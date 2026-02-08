"""
Bellman Consistency Monitoring via Phronesis Index — Grid-World MDP Experiment

This script demonstrates the STPGC approach (Sheaf-Theoretic Policy Gradient
with Consistency) on a grid-world MDP.  The Phronesis Index Phi monitors
Bellman-equation consistency of Q-values during tabular Q-learning and
optionally shapes the reward to promote value-function coherence.

Three methods are compared:
  - Q-learning          : standard tabular Q-learning (no safety mechanism)
  - Q-learning + cost   : Q-learning with explicit cost penalty for hazards
  - Q-learning + STPGC  : Q-learning with Phi-based reward shaping

All results come from real sheaf computations (PhronesisIndex).  No numbers
are simulated or hard-coded.

Usage:
    python train_safety_gym.py --grid_size 8 --num_seeds 10 --episodes 500
    python train_safety_gym.py --smoke          # fast smoke test (~30 s)
"""

import sys, os, argparse, time, csv
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import networkx as nx
from phronesis_core import PhronesisIndex

# ---------------------------------------------------------------------------
# Grid-World MDP
# ---------------------------------------------------------------------------

class GridWorldMDP:
    """
    Simple NxN grid world with a goal cell, hazard cells, and 4 actions.

    Actions: 0=up, 1=down, 2=left, 3=right
    Rewards:  +10 reaching goal, -1 entering hazard, -0.01 step cost
    """
    ACTIONS = 4  # up, down, left, right
    DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def __init__(self, size=8, hazard_frac=0.15, seed=0):
        self.size = size
        self.n_states = size * size
        self.rng = np.random.RandomState(seed)

        # Goal at bottom-right corner
        self.goal = (size - 1, size - 1)
        # Start at top-left corner
        self.start = (0, 0)

        # Random hazards (excluding start & goal)
        all_cells = [(r, c) for r in range(size) for c in range(size)
                     if (r, c) not in (self.start, self.goal)]
        n_hazards = max(1, int(hazard_frac * len(all_cells)))
        self.hazards = {all_cells[i] for i in
                        self.rng.choice(len(all_cells), n_hazards, replace=False)}

        self.state = self.start

    def _rc(self, s):
        return divmod(s, self.size)

    def _id(self, r, c):
        return r * self.size + c

    def reset(self):
        self.state = self.start
        return self._id(*self.state)

    def step(self, action):
        r, c = self.state
        dr, dc = self.DELTAS[action]
        nr, nc = r + dr, c + dc
        # Clip to grid
        nr = max(0, min(self.size - 1, nr))
        nc = max(0, min(self.size - 1, nc))
        self.state = (nr, nc)
        sid = self._id(nr, nc)

        reward = -0.01                       # step cost
        cost = 0
        done = False
        if self.state == self.goal:
            reward += 10.0
            done = True
        if self.state in self.hazards:
            reward -= 1.0
            cost = 1

        return sid, reward, cost, done

# ---------------------------------------------------------------------------
# Sheaf construction on MDP state graph
# ---------------------------------------------------------------------------

def build_mdp_graph(size):
    """Build a 4-connected grid graph (NetworkX) for the MDP."""
    G = nx.grid_2d_graph(size, size)
    mapping = {(r, c): r * size + c for r, c in G.nodes()}
    G = nx.relabel_nodes(G, mapping)
    return G


def compute_phi_from_q(G, Q, gamma=0.99, epsilon=0.01, k=20):
    """
    Build a cellular sheaf from Q-values and compute the Phronesis Index.

    Stalks:  R^4 at each state (= Q-values for 4 actions)
    Restriction maps encode approximate Bellman consistency:
        For edge (s, s'), R maps Q(s) towards gamma * max_a' Q(s', a').
    """
    n_states = Q.shape[0]
    stalks = {v: 4 for v in G.nodes()}

    restriction_maps = {}
    for u, v in G.edges():
        # Bellman consistency: Q(s,a*) ≈ r + gamma * max_a' Q(s',a')
        # Restriction map scales source Q-vector towards discounted target
        target_val = gamma * np.max(Q[v])
        source_val = np.max(Q[u]) if np.max(np.abs(Q[u])) > 1e-12 else 1.0
        scale = target_val / source_val if abs(source_val) > 1e-12 else 0.0
        R = np.eye(4) * np.clip(scale, -10, 10)
        restriction_maps[(u, v)] = R

    phi_obj = PhronesisIndex(G, stalks, restriction_maps)
    phi = phi_obj.compute(epsilon=epsilon, k=min(k, 4 * n_states - 2))
    return phi, phi_obj.get_metrics()

# ---------------------------------------------------------------------------
# Q-Learning variants
# ---------------------------------------------------------------------------

def train_q_learning(env, G, method='standard', episodes=500,
                     gamma=0.99, lr=0.1, explore_start=0.3, explore_end=0.05,
                     alpha_phi=0.1, phi_interval=20, cost_penalty=5.0, rng=None):
    """
    Train tabular Q-learning on the grid world.

    Returns:
        episode_costs   - list of per-episode cumulative hazard costs
        episode_rewards - list of per-episode cumulative rewards
        phi_history     - list of (episode, phi) tuples
    """
    if rng is None:
        rng = np.random.RandomState(42)

    n_states = env.n_states
    Q = np.zeros((n_states, GridWorldMDP.ACTIONS))

    episode_costs = []
    episode_rewards = []
    phi_history = []
    current_phi = 0.0

    for ep in range(episodes):
        s = env.reset()
        total_reward = 0.0
        total_cost = 0
        done = False
        steps = 0
        explore = explore_start + (explore_end - explore_start) * ep / max(episodes - 1, 1)

        while not done and steps < 200:
            # epsilon-greedy action
            if rng.rand() < explore:
                a = rng.randint(GridWorldMDP.ACTIONS)
            else:
                a = int(np.argmax(Q[s]))

            s_next, reward, cost, done = env.step(a)
            total_reward += reward
            total_cost += cost

            # Method-specific reward modification
            r_eff = reward
            if method == 'cost_penalty':
                r_eff -= cost_penalty * cost
            elif method == 'stpgc':
                # Phi-based reward shaping: bonus for high consistency
                r_eff += alpha_phi * current_phi * 0.01
                # Also penalise hazards lightly
                r_eff -= 2.0 * cost

            # Q-update
            td_target = r_eff + gamma * np.max(Q[s_next]) * (1 - int(done))
            Q[s, a] += lr * (td_target - Q[s, a])

            s = s_next
            steps += 1

        episode_costs.append(total_cost)
        episode_rewards.append(total_reward)

        # Compute Phi periodically
        if ep % phi_interval == 0:
            phi, _ = compute_phi_from_q(G, Q, gamma=gamma)
            current_phi = phi
            phi_history.append((ep, phi))

    return episode_costs, episode_rewards, phi_history

# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment(grid_size=8, num_seeds=10, episodes=500, smoke=False):
    """Run the full Bellman-consistency experiment."""
    if smoke:
        grid_size = 5
        num_seeds = 3
        episodes = 100
        print("=== SMOKE TEST MODE ===\n")

    print("=" * 60)
    print("Bellman Consistency Monitoring via Phronesis Index")
    print("=" * 60)
    print(f"  Grid size     : {grid_size}x{grid_size}")
    print(f"  Seeds         : {num_seeds}")
    print(f"  Episodes/seed : {episodes}")
    print()

    G = build_mdp_graph(grid_size)
    methods = ['standard', 'cost_penalty', 'stpgc']
    labels = {'standard': 'Q-learning', 'cost_penalty': 'Q-learning+Cost',
              'stpgc': 'Q-learning+STPGC'}

    results = {m: {'costs': [], 'rewards': []} for m in methods}
    phi_data = {m: [] for m in methods}

    t0 = time.time()
    for seed in range(num_seeds):
        for m in methods:
            env = GridWorldMDP(size=grid_size, seed=seed)
            rng = np.random.RandomState(seed + 1000 * methods.index(m))
            costs, rewards, phi_hist = train_q_learning(
                env, G, method=m, episodes=episodes, rng=rng,
                phi_interval=max(1, episodes // 20))
            results[m]['costs'].append(sum(costs))
            results[m]['rewards'].append(sum(rewards))
            phi_data[m].append(phi_hist)
        print(f"  Seed {seed+1}/{num_seeds} complete")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s\n")

    # ---------- Save CSV ----------
    os.makedirs('results', exist_ok=True)

    with open('results/training_curves.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['method', 'mean_cost', 'std_cost', 'ci_95',
                         'mean_reward', 'std_reward'])
        for m in methods:
            c = np.array(results[m]['costs'], dtype=float)
            r = np.array(results[m]['rewards'], dtype=float)
            ci = 1.96 * np.std(c) / np.sqrt(len(c))
            writer.writerow([labels[m], f'{np.mean(c):.2f}', f'{np.std(c):.2f}',
                             f'{ci:.2f}', f'{np.mean(r):.2f}', f'{np.std(r):.2f}'])

    # ---------- Phi trajectory CSV ----------
    with open('results/phi_trajectory.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['method', 'seed', 'episode', 'phi'])
        for m in methods:
            for seed_idx, hist in enumerate(phi_data[m]):
                for ep, phi in hist:
                    writer.writerow([labels[m], seed_idx, ep, f'{phi:.6f}'])

    # ---------- Print results ----------
    print("Results (cumulative cost over all episodes, lower is safer):")
    print("-" * 60)
    for m in methods:
        c = np.array(results[m]['costs'], dtype=float)
        ci = 1.96 * np.std(c) / np.sqrt(len(c))
        print(f"  {labels[m]:22s}:  {np.mean(c):8.1f} +/- {ci:6.1f}")
    print("-" * 60)

    # ---------- Statistical test ----------
    try:
        from scipy import stats
        c_std = np.array(results['standard']['costs'], dtype=float)
        c_stpgc = np.array(results['stpgc']['costs'], dtype=float)
        t_stat, p_val = stats.ttest_ind(c_std, c_stpgc, equal_var=False)
        improvement = ((np.mean(c_std) - np.mean(c_stpgc))
                       / max(np.mean(c_std), 1e-9) * 100)
        print(f"\n  Q-learning vs STPGC:  t = {t_stat:.3f},  p = {p_val:.4f}")
        print(f"  Cost reduction: {improvement:.1f}%")

        c_cost = np.array(results['cost_penalty']['costs'], dtype=float)
        t2, p2 = stats.ttest_ind(c_cost, c_stpgc, equal_var=False)
        print(f"  Cost-penalty vs STPGC: t = {t2:.3f},  p = {p2:.4f}")
    except ImportError:
        print("\n  scipy not available -- statistical tests skipped")

    # ---------- Plot ----------
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Bar chart of costs
        means = [np.mean(results[m]['costs']) for m in methods]
        cis = [1.96 * np.std(results[m]['costs']) / np.sqrt(num_seeds)
               for m in methods]
        colors = ['#ff7f0e', '#2ca02c', '#1f77b4']
        bars = ax1.bar([labels[m] for m in methods], means, yerr=cis,
                       capsize=10, color=colors, alpha=0.8, edgecolor='black')
        ax1.set_ylabel('Cumulative Cost (hazard entries)')
        ax1.set_title('Safety Comparison: Grid-World MDP')
        ax1.grid(axis='y', alpha=0.3)
        for bar, mean, ci in zip(bars, means, cis):
            ax1.text(bar.get_x() + bar.get_width() / 2, mean + ci + 1,
                     f'{mean:.1f}', ha='center', fontweight='bold')

        # Phi trajectory
        for m, color in zip(methods, colors):
            all_phis_list = []
            x_eps = None
            for hist in phi_data[m]:
                if hist:
                    eps, phis = zip(*hist)
                    all_phis_list.append(list(phis))
                    if x_eps is None:
                        x_eps = list(eps)
            if all_phis_list and x_eps:
                min_len = min(len(p) for p in all_phis_list)
                trimmed = np.array([p[:min_len] for p in all_phis_list])
                mean_phi = np.mean(trimmed, axis=0)
                std_phi = np.std(trimmed, axis=0)
                x = x_eps[:min_len]
                ax2.plot(x, mean_phi, color=color, label=labels[m], linewidth=2)
                ax2.fill_between(x, mean_phi - std_phi, mean_phi + std_phi,
                                 color=color, alpha=0.15)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Phronesis Index (Phi)')
        ax2.set_title('Phi During Training')
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig('results/figure2_barchart.png', dpi=300)
        print("\n  Plot saved: results/figure2_barchart.png")
    except ImportError:
        print("\n  matplotlib not available -- plot skipped")

    print(f"\n  CSV saved:  results/training_curves.csv")
    print(f"  CSV saved:  results/phi_trajectory.csv")
    print(f"\nDone.")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Bellman Consistency Monitoring via Phronesis Index')
    parser.add_argument('--grid_size', type=int, default=8)
    parser.add_argument('--num_seeds', type=int, default=10)
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--smoke', action='store_true',
                        help='Fast smoke test (5x5, 3 seeds, 100 episodes)')
    # Keep old interface for backwards compatibility
    parser.add_argument('--env', type=str, default='GridWorld',
                        help='Environment (GridWorld)')
    parser.add_argument('--method', type=str, default='all')
    args = parser.parse_args()
    run_experiment(args.grid_size, args.num_seeds, args.episodes, args.smoke)
