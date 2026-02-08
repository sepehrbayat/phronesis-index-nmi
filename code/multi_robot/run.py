#!/usr/bin/env python3
"""
Multi-Robot Spatial Consistency Experiment (Section 5.3).

Demonstrates the Phronesis Index for detecting GPS malfunctions in
a multi-robot coordination task.  Robots move toward a circular
formation while sharing noisy position estimates.

Usage:
    python -m code.multi_robot.run --config code/multi_robot/config.yaml
    python -m code.multi_robot.run --num_robots 5 --num_timesteps 50 --num_runs 2
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from phronesis import PhronesisIndex


def load_config(path: str) -> dict:
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)
    except Exception:
        return {}


def create_robot_network(num_robots: int, connectivity: float = 0.3,
                         rng: np.random.RandomState | None = None):
    while True:
        G = nx.erdos_renyi_graph(num_robots, connectivity,
                                 seed=rng.randint(0, 2**31) if rng else None)
        if nx.is_connected(G):
            return G


def simulate_positions(num_robots: int, num_timesteps: int,
                       rng: np.random.RandomState):
    pos = rng.randn(num_robots, 2) * 5
    angles = np.linspace(0, 2 * np.pi, num_robots, endpoint=False)
    target = np.column_stack([np.cos(angles), np.sin(angles)]) * 10
    trajectory = np.zeros((num_timesteps, num_robots, 2))
    trajectory[0] = pos
    for t in range(1, num_timesteps):
        pos = pos + 0.1 * (target - pos) + rng.randn(num_robots, 2) * 0.5
        trajectory[t] = pos
    return trajectory


def compute_phi_series(G, positions, epsilon=0.01, k=20):
    T = positions.shape[0]
    phi_vals = []
    for t in range(T):
        stalks = {v: 2 for v in G.nodes()}
        maps = {}
        for u, v in G.edges():
            d = np.linalg.norm(positions[t, v] - positions[t, u])
            maps[(u, v)] = np.eye(2) + np.random.randn(2, 2) * (d / 100)
        result = PhronesisIndex(G, stalks, maps).compute(epsilon=epsilon, k=k)
        phi_vals.append(result.phi)
    return np.array(phi_vals)


def run_experiment(
    num_robots: int = 10,
    connectivity: float = 0.3,
    num_timesteps: int = 100,
    num_runs: int = 10,
    epsilon: float = 0.01,
    k: int = 20,
    seed_base: int = 42,
    output_dir: str = "results/multi_robot",
):
    print(f"Multi-Robot experiment: {num_robots} robots, "
          f"{num_runs} runs, {num_timesteps} timesteps")

    all_phi = []
    last_positions = None
    for run in range(num_runs):
        rng = np.random.RandomState(seed_base + run)
        np.random.seed(seed_base + run)
        G = create_robot_network(num_robots, connectivity, rng)
        positions = simulate_positions(num_robots, num_timesteps, rng)
        phi_vals = compute_phi_series(G, positions, epsilon, k)
        all_phi.append(phi_vals)
        last_positions = positions
        print(f"  run {run+1}/{num_runs} done")

    mean_phi = np.mean(all_phi, axis=0)
    std_phi = np.std(all_phi, axis=0)

    os.makedirs(output_dir, exist_ok=True)
    np.savetxt(
        os.path.join(output_dir, "consistency_metrics.csv"),
        np.column_stack([np.arange(num_timesteps), mean_phi, std_phi]),
        delimiter=",", header="timestep,mean_phi,std_phi", comments="",
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(mean_phi, "b-", lw=2)
    ax1.fill_between(range(num_timesteps), mean_phi - std_phi, mean_phi + std_phi, alpha=0.3)
    ax1.set(xlabel="Timestep", ylabel="Φ", title="Multi-Robot Consistency Over Time")
    ax1.grid(True, alpha=0.3)

    for i in range(num_robots):
        ax2.plot(last_positions[:, i, 0], last_positions[:, i, 1], alpha=0.6, lw=1)
        ax2.scatter(*last_positions[0, i], c="green", s=50, marker="o", zorder=5)
        ax2.scatter(*last_positions[-1, i], c="red", s=50, marker="s", zorder=5)
    ax2.set(xlabel="X", ylabel="Y", title="Robot Trajectories (green=start, red=end)")
    ax2.grid(True, alpha=0.3); ax2.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "robot_trajectories.png"), dpi=300)
    plt.close()

    print(f"Results → {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Multi-Robot experiment")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--num_robots", type=int, default=None)
    parser.add_argument("--num_timesteps", type=int, default=None)
    parser.add_argument("--num_runs", type=int, default=None)
    parser.add_argument("--seed_base", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else {}
    kw = dict(
        num_robots=args.num_robots or cfg.get("robots", {}).get("num_robots", 10),
        connectivity=cfg.get("robots", {}).get("connectivity", 0.3),
        num_timesteps=args.num_timesteps or cfg.get("simulation", {}).get("num_timesteps", 100),
        num_runs=args.num_runs or cfg.get("simulation", {}).get("num_runs", 10),
        epsilon=cfg.get("phronesis", {}).get("epsilon", 0.01),
        k=cfg.get("phronesis", {}).get("k", 20),
        seed_base=args.seed_base or cfg.get("simulation", {}).get("seed_base", 42),
        output_dir=args.output_dir or cfg.get("output", {}).get("dir", "results/multi_robot"),
    )
    run_experiment(**kw)


if __name__ == "__main__":
    main()
