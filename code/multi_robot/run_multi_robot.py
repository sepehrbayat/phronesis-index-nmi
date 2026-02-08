"""
Multi-Robot Spatial Consistency Experiment

This script demonstrates the Phronesis Index for detecting inconsistencies
in multi-robot coordination tasks.
"""

import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
from phronesis_core import PhronesisIndex
import networkx as nx
import argparse
import os


def create_robot_network(num_robots, connectivity=0.3):
    """Create a random communication network for robots."""
    G = nx.erdos_renyi_graph(num_robots, connectivity)
    # Ensure connectivity
    while not nx.is_connected(G):
        G = nx.erdos_renyi_graph(num_robots, connectivity)
    return G


def simulate_robot_positions(num_robots, num_timesteps):
    """
    Simulate robot positions over time.
    Robots move in 2D space and try to maintain formation.
    """
    positions = np.zeros((num_timesteps, num_robots, 2))
    
    # Initialize random positions
    positions[0] = np.random.randn(num_robots, 2) * 5
    
    # Target formation (circle)
    angles = np.linspace(0, 2*np.pi, num_robots, endpoint=False)
    target_positions = np.column_stack([np.cos(angles), np.sin(angles)]) * 10
    
    # Simulate motion
    for t in range(1, num_timesteps):
        # Move towards target with noise
        positions[t] = positions[t-1] + 0.1 * (target_positions - positions[t-1])
        positions[t] += np.random.randn(num_robots, 2) * 0.5
    
    return positions


def compute_consistency_over_time(G, positions):
    """
    Compute Phronesis Index over time based on robot positions.
    """
    num_timesteps = positions.shape[0]
    phi_values = []
    
    for t in range(num_timesteps):
        # Create sheaf structure based on current positions
        stalks = {v: 2 for v in G.nodes()}  # 2D position for each robot
        
        # Restriction maps based on relative positions
        restriction_maps = {}
        for u, v in G.edges():
            # Compute relative position
            rel_pos = positions[t, v] - positions[t, u]
            # Create restriction map (should preserve relative position)
            # For simplicity, use identity with small perturbation based on distance
            distance = np.linalg.norm(rel_pos)
            perturbation = np.random.randn(2, 2) * (distance / 10)
            restriction_maps[(u, v)] = np.eye(2) + perturbation * 0.1
        
        # Compute Phronesis Index
        phi_obj = PhronesisIndex(G, stalks, restriction_maps)
        phi = phi_obj.compute(epsilon=0.01, k=20)
        phi_values.append(phi)
    
    return np.array(phi_values)


def run_experiment(num_robots=10, num_timesteps=100, num_runs=10):
    """
    Run the multi-robot coordination experiment.
    
    Parameters:
    - num_robots: Number of robots in the system
    - num_timesteps: Number of timesteps to simulate
    - num_runs: Number of independent runs
    """
    print(f"Running Multi-Robot Coordination experiment:")
    print(f"  Number of robots: {num_robots}")
    print(f"  Timesteps: {num_timesteps}")
    print(f"  Runs: {num_runs}")
    print()
    
    all_phi_values = []
    
    for run in range(num_runs):
        print(f"Run {run + 1}/{num_runs}...", end=" ")
        
        # Create robot network
        G = create_robot_network(num_robots)
        
        # Simulate robot positions
        positions = simulate_robot_positions(num_robots, num_timesteps)
        
        # Compute consistency over time
        phi_values = compute_consistency_over_time(G, positions)
        all_phi_values.append(phi_values)
        
        print("Done")
    
    # Average across runs
    mean_phi = np.mean(all_phi_values, axis=0)
    std_phi = np.std(all_phi_values, axis=0)
    
    # Save results
    os.makedirs('results', exist_ok=True)
    
    np.savetxt('results/consistency_metrics.csv',
               np.column_stack([np.arange(num_timesteps), mean_phi, std_phi]),
               delimiter=',',
               header='timestep,mean_phi,std_phi',
               comments='')
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Consistency over time
    ax1.plot(mean_phi, 'b-', linewidth=2)
    ax1.fill_between(range(num_timesteps),
                     mean_phi - std_phi,
                     mean_phi + std_phi,
                     alpha=0.3)
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Phronesis Index (Φ)')
    ax1.set_title('Multi-Robot Consistency Over Time')
    ax1.grid(True, alpha=0.3)
    
    # Robot trajectories (last run)
    positions = simulate_robot_positions(num_robots, num_timesteps)
    for i in range(num_robots):
        ax2.plot(positions[:, i, 0], positions[:, i, 1], alpha=0.6, linewidth=1)
        ax2.scatter(positions[0, i, 0], positions[0, i, 1], c='green', s=50, marker='o')
        ax2.scatter(positions[-1, i, 0], positions[-1, i, 1], c='red', s=50, marker='s')
    
    ax2.set_xlabel('X Position')
    ax2.set_ylabel('Y Position')
    ax2.set_title('Robot Trajectories (Green=Start, Red=End)')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    plt.tight_layout()
    plt.savefig('results/robot_trajectories.png', dpi=300)
    
    print(f"\nResults saved to results/")
    print(f"  - consistency_metrics.csv")
    print(f"  - robot_trajectories.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Multi-Robot experiment')
    parser.add_argument('--num_robots', type=int, default=10, help='Number of robots')
    parser.add_argument('--num_timesteps', type=int, default=100, help='Number of timesteps')
    parser.add_argument('--num_runs', type=int, default=10, help='Number of runs')
    
    args = parser.parse_args()
    
    run_experiment(args.num_robots, args.num_timesteps, args.num_runs)
