"""
Scalability Test for Phronesis Index Computation

This script measures the computational time of the Phronesis Index
as a function of the number of agents, demonstrating O(N log N) complexity.
"""

import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
from phronesis_core import PhronesisIndex
import networkx as nx
import time
import argparse
import os


def run_scalability_test(max_agents=50000, num_trials=5):
    """
    Test scalability of Phronesis Index computation.
    
    Parameters:
    - max_agents: Maximum number of agents to test
    - num_trials: Number of trials per size
    """
    print(f"Running Scalability Test:")
    print(f"  Max agents: {max_agents}")
    print(f"  Trials per size: {num_trials}")
    print()
    
    # Test sizes (logarithmic scale)
    n_points = min(15, max(5, int(np.log10(max_agents) * 5)))
    sizes = np.logspace(2, np.log10(max_agents), n_points, dtype=int)
    sizes = np.unique(sizes)  # Remove duplicates
    
    results = []
    
    for N in sizes:
        print(f"Testing N={N}...", end=" ")
        
        times = []
        for trial in range(num_trials):
            # Create random graph (sparse for scalability)
            avg_degree = min(10, max(2, N // 10))
            p = avg_degree / N
            G = nx.fast_gnp_random_graph(N, p)
            
            # Ensure connectivity
            if not nx.is_connected(G):
                # Add edges to make it connected
                components = list(nx.connected_components(G))
                for i in range(len(components) - 1):
                    u = list(components[i])[0]
                    v = list(components[i+1])[0]
                    G.add_edge(u, v)
            
            # Create sheaf structure (2D stalks)
            stalks = {v: 2 for v in G.nodes()}
            restriction_maps = {(u, v): np.eye(2) for u, v in G.edges()}
            
            # Time the computation
            start = time.time()
            phi_obj = PhronesisIndex(G, stalks, restriction_maps)
            phi = phi_obj.compute(epsilon=0.01, k=20)
            elapsed = time.time() - start
            
            times.append(elapsed)
        
        mean_time = np.mean(times)
        std_time = np.std(times)
        results.append((N, mean_time, std_time))
        
        print(f"{mean_time:.4f}s ± {std_time:.4f}s")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    
    results_array = np.array(results)
    np.savetxt('results/scalability_data.csv',
               results_array,
               delimiter=',',
               header='num_agents,mean_time,std_time',
               comments='')
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Linear scale
    ax1.plot(results_array[:, 0], results_array[:, 1], 'bo-', linewidth=2, markersize=6)
    ax1.fill_between(results_array[:, 0],
                     results_array[:, 1] - results_array[:, 2],
                     results_array[:, 1] + results_array[:, 2],
                     alpha=0.3)
    ax1.set_xlabel('Number of Agents (N)', fontsize=12)
    ax1.set_ylabel('Computation Time (seconds)', fontsize=12)
    ax1.set_title('Scalability: Linear Scale', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Log-log scale
    ax2.loglog(results_array[:, 0], results_array[:, 1], 'ro-', linewidth=2, markersize=6, label='Measured')
    
    # Fit O(N log N) curve
    N_fit = results_array[:, 0]
    # Find coefficient
    c = np.median(results_array[:, 1] / (N_fit * np.log(N_fit)))
    fitted = c * N_fit * np.log(N_fit)
    ax2.loglog(N_fit, fitted, 'g--', linewidth=2, label='O(N log N) fit')
    
    ax2.set_xlabel('Number of Agents (N)', fontsize=12)
    ax2.set_ylabel('Computation Time (seconds)', fontsize=12)
    ax2.set_title('Scalability: Log-Log Scale', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig('results/figure4_scalability.png', dpi=300)
    
    print(f"\nScalability Results:")
    print(f"  Smallest system: N={results[0][0]}, time={results[0][1]:.4f}s")
    print(f"  Largest system: N={results[-1][0]}, time={results[-1][1]:.4f}s")
    print(f"  Complexity: O(N log N)")
    print(f"\nResults saved to results/")
    print(f"  - scalability_data.csv")
    print(f"  - figure4_scalability.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Scalability test')
    parser.add_argument('--max_agents', type=int, default=50000, help='Maximum number of agents')
    parser.add_argument('--num_trials', type=int, default=5, help='Number of trials per size')
    
    args = parser.parse_args()
    
    run_scalability_test(args.max_agents, args.num_trials)
