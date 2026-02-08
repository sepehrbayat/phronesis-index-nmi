"""
Logic Maze Anomaly Detection Experiment

This script implements the Logic Maze experiment from the paper, demonstrating
how the Phronesis Index detects anomalies in a multi-agent navigation task.
"""

import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
from phronesis_core import PhronesisIndex
import networkx as nx
import argparse


def create_grid_graph(grid_size):
    """Create a grid graph for the Logic Maze."""
    G = nx.grid_2d_graph(grid_size, grid_size)
    # Convert to simple graph with integer node IDs
    mapping = {node: i for i, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)
    return G


def create_sheaf_structure(G, grid_size):
    """
    Create cellular sheaf structure for Logic Maze.
    Each agent has a 2D orientation vector (stalk dimension = 2).
    """
    stalks = {v: 2 for v in G.nodes()}
    
    # Restriction maps enforce orientation consistency
    restriction_maps = {}
    for u, v in G.edges():
        # Identity map (agents should have consistent orientations)
        restriction_maps[(u, v)] = np.eye(2)
    
    return stalks, restriction_maps


def inject_anomaly(restriction_maps, anomaly_edges, anomaly_strength=0.5):
    """
    Inject anomalies by perturbing restriction maps on specific edges.
    """
    perturbed_maps = restriction_maps.copy()
    for edge in anomaly_edges:
        if edge in perturbed_maps:
            # Add random perturbation
            perturbation = np.random.randn(2, 2) * anomaly_strength
            perturbed_maps[edge] = perturbed_maps[edge] + perturbation
    return perturbed_maps


def run_experiment(grid_size=5, anomaly_time=50, num_runs=10, num_timesteps=150):
    """
    Run the Logic Maze experiment.
    
    Parameters:
    - grid_size: Size of the grid (grid_size x grid_size agents)
    - anomaly_time: Timestep at which anomaly is injected
    - num_runs: Number of independent runs
    - num_timesteps: Total number of timesteps
    """
    print(f"Running Logic Maze experiment:")
    print(f"  Grid size: {grid_size}x{grid_size}")
    print(f"  Anomaly injection time: {anomaly_time}")
    print(f"  Number of runs: {num_runs}")
    print(f"  Total timesteps: {num_timesteps}")
    print()
    
    # Store results
    all_phi_values = []
    
    for run in range(num_runs):
        print(f"Run {run + 1}/{num_runs}...", end=" ")
        
        # Create graph and sheaf structure
        G = create_grid_graph(grid_size)
        stalks, restriction_maps = create_sheaf_structure(G, grid_size)
        
        # Time series of Phronesis Index
        phi_timeseries = []
        
        for t in range(num_timesteps):
            # Inject anomaly at specified time
            if anomaly_time <= t < anomaly_time + 50:
                # Select random edges for anomaly
                edges_list = list(G.edges())
                num_anomaly_edges = max(1, len(edges_list) // 10)
                idx = np.random.choice(len(edges_list), num_anomaly_edges, replace=False)
                anomaly_edges = [edges_list[i] for i in idx]
                current_maps = inject_anomaly(restriction_maps, anomaly_edges)
            else:
                current_maps = restriction_maps
            
            # Compute Phronesis Index
            phi_obj = PhronesisIndex(G, stalks, current_maps)
            phi = phi_obj.compute(epsilon=0.01, k=20)
            phi_timeseries.append(phi)
        
        all_phi_values.append(phi_timeseries)
        print("Done")
    
    # Average across runs
    mean_phi = np.mean(all_phi_values, axis=0)
    std_phi = np.std(all_phi_values, axis=0)
    
    # Save results
    import os
    os.makedirs('results', exist_ok=True)
    
    np.savetxt('results/phi_timeseries.csv', 
               np.column_stack([np.arange(num_timesteps), mean_phi, std_phi]),
               delimiter=',',
               header='timestep,mean_phi,std_phi',
               comments='')
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(mean_phi, 'b-', linewidth=2, label='Phronesis Index')
    plt.fill_between(range(num_timesteps), 
                     mean_phi - std_phi, 
                     mean_phi + std_phi, 
                     alpha=0.3)
    plt.axvline(x=anomaly_time, color='r', linestyle='--', label='Anomaly injected')
    plt.axvline(x=anomaly_time + 50, color='g', linestyle='--', label='Anomaly removed')
    plt.xlabel('Timestep')
    plt.ylabel('Phronesis Index (Φ)')
    plt.title('Logic Maze: Anomaly Detection via Phronesis Index')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/figure1_timeseries.png', dpi=300)
    print(f"\nResults saved to results/")
    print(f"  - phi_timeseries.csv")
    print(f"  - figure1_timeseries.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Logic Maze experiment')
    parser.add_argument('--grid_size', type=int, default=5, help='Grid size')
    parser.add_argument('--anomaly_time', type=int, default=50, help='Anomaly injection time')
    parser.add_argument('--num_runs', type=int, default=10, help='Number of runs')
    parser.add_argument('--num_timesteps', type=int, default=150, help='Total timesteps')
    
    args = parser.parse_args()
    
    run_experiment(args.grid_size, args.anomaly_time, args.num_runs, args.num_timesteps)
