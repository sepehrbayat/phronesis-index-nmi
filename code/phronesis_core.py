"""
Core implementation of the Phronesis Index for consistency detection in multi-agent systems.

This module provides the PhronesisIndex class for computing the spectral-topological
consistency metric based on cellular sheaf theory.
"""

import numpy as np
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import eigsh
import networkx as nx


class PhronesisIndex:
    """
    Compute the Phronesis Index for a multi-agent system represented as a cellular sheaf.
    
    The Phronesis Index Φ = λ₁⁺ / (h¹_ε + ε) combines:
    - λ₁⁺: the smallest positive eigenvalue of the Connection Laplacian (spectral gap)
    - h¹_ε: the ε-approximate first cohomology dimension (number of near-zero eigenvalues - 1)
    - ε: a small regularization parameter
    
    Parameters
    ----------
    graph : networkx.Graph
        The communication graph where vertices represent agents and edges represent
        communication links.
    stalks : dict
        Dictionary mapping vertex IDs to stalk dimensions (int). Each stalk is a
        vector space R^d attached to a vertex.
    restriction_maps : dict
        Dictionary mapping edge tuples (u, v) to restriction map matrices (numpy arrays).
        Each restriction map R_{uv} is a d_v × d_u matrix that enforces consistency
        constraints between stalks at vertices u and v.
    """
    
    def __init__(self, graph, stalks, restriction_maps):
        self.graph = graph
        self.stalks = stalks
        self.restriction_maps = restriction_maps
        self.N = graph.number_of_nodes()
        self.M = graph.number_of_edges()
        
        # Compute dim(H^0) = n_connected_components * stalk_dimension
        # For a connected graph with d-dimensional constant stalks, dim(H^0) = d
        n_comp = nx.number_connected_components(graph)
        stalk_dims = set(stalks.values())
        d = max(stalk_dims) if stalk_dims else 1
        self._dim_h0 = n_comp * d
        
        # Build the Connection Laplacian
        self._build_laplacian()
    
    def _build_laplacian(self):
        """
        Construct the Connection Laplacian matrix L = B^T B, where B is the
        boundary operator (coboundary map) of the cellular sheaf.
        """
        # Compute total dimension of all stalks
        nodes = list(self.graph.nodes())
        total_dim = sum(self.stalks[v] for v in nodes)
        
        # Pre-compute vertex-to-index mapping and stalk offsets (O(N) once)
        vertex_to_idx = {v: i for i, v in enumerate(nodes)}
        offsets = np.zeros(len(nodes) + 1, dtype=int)
        for i, v in enumerate(nodes):
            offsets[i + 1] = offsets[i] + self.stalks[v]
        
        # Build the boundary operator B (sparse matrix)
        # B maps from vertex sections to edge sections
        rows, cols, data = [], [], []
        
        edge_offset = 0
        for u, v in self.graph.edges():
            d_u = self.stalks[u]
            d_v = self.stalks[v]
            R_uv = self.restriction_maps.get((u, v), np.eye(d_v, d_u))
            
            # Get global offsets for stalks (O(1) lookup)
            u_start = offsets[vertex_to_idx[u]]
            v_start = offsets[vertex_to_idx[v]]
            
            # Add entries for R_uv (from u) and -I (from v)
            for i in range(d_v):
                for j in range(d_u):
                    if R_uv[i, j] != 0:
                        rows.append(edge_offset + i)
                        cols.append(u_start + j)
                        data.append(R_uv[i, j])
                
                rows.append(edge_offset + i)
                cols.append(v_start + i)
                data.append(-1.0)
            
            edge_offset += d_v
        
        # Build sparse boundary matrix
        B = csr_matrix((data, (rows, cols)), shape=(edge_offset, total_dim))
        
        # Connection Laplacian L = B^T B
        self.laplacian = B.T @ B
        self.total_dim = total_dim
    
    def compute(self, epsilon=0.01, k=20):
        """
        Compute the Phronesis Index.
        
        Parameters
        ----------
        epsilon : float, optional
            Threshold for counting near-zero eigenvalues. Default is 0.01.
        k : int, optional
            Number of smallest eigenvalues to compute. Default is 20.
        
        Returns
        -------
        phi : float
            The Phronesis Index value.
        """
        # Compute k smallest eigenvalues
        eigenvalues = eigsh(self.laplacian, k=min(k, self.total_dim - 1), 
                           which='SM', return_eigenvectors=False)
        eigenvalues = np.sort(eigenvalues)
        
        # Find the smallest positive eigenvalue (spectral gap)
        positive_eigs = eigenvalues[eigenvalues > 1e-8]
        if len(positive_eigs) == 0:
            lambda_1_plus = epsilon  # Degenerate case
        else:
            lambda_1_plus = positive_eigs[0]
        
        # Count near-zero eigenvalues (approximate cohomology dimension)
        # h1_epsilon = #{lambda_i < epsilon} - dim(H^0)
        # dim(H^0) = n_components * d for d-dimensional constant stalks
        n_near_zero = int(np.sum(eigenvalues < epsilon))
        h1_epsilon = max(n_near_zero - self._dim_h0, 0)
        
        # Compute Phronesis Index
        phi = lambda_1_plus / (h1_epsilon + epsilon)
        
        # Store metrics
        self.metrics = {
            'phi': phi,
            'lambda_1_plus': lambda_1_plus,
            'h1_epsilon': h1_epsilon,
            'dim_h0': self._dim_h0,
            'epsilon': epsilon,
            'eigenvalues': eigenvalues
        }
        
        return phi
    
    def get_metrics(self):
        """
        Get detailed metrics from the last computation.
        
        Returns
        -------
        metrics : dict
            Dictionary containing:
            - 'phi': Phronesis Index value
            - 'lambda_1_plus': smallest positive eigenvalue
            - 'h1_epsilon': approximate cohomology dimension
            - 'epsilon': threshold used
            - 'eigenvalues': array of computed eigenvalues
        """
        if not hasattr(self, 'metrics'):
            raise ValueError("Must call compute() before get_metrics()")
        return self.metrics


def example_usage():
    """
    Example: Compute Phronesis Index for a simple 3-agent system.
    """
    # Create a simple graph: 3 agents in a triangle
    G = nx.cycle_graph(3)
    
    # Each agent has a 2D stalk (e.g., position in 2D space)
    stalks = {0: 2, 1: 2, 2: 2}
    
    # Restriction maps enforce equality (identity matrices)
    restriction_maps = {
        (0, 1): np.eye(2),
        (1, 2): np.eye(2),
        (2, 0): np.eye(2)
    }
    
    # Create PhronesisIndex object
    phi_obj = PhronesisIndex(G, stalks, restriction_maps)
    
    # Compute the index
    phi = phi_obj.compute(epsilon=0.01)
    
    print(f"Phronesis Index: {phi:.4f}")
    
    # Get detailed metrics
    metrics = phi_obj.get_metrics()
    print(f"Spectral gap (λ₁⁺): {metrics['lambda_1_plus']:.4f}")
    print(f"Cohomology dimension (h¹_ε): {metrics['h1_epsilon']}")
    print(f"Eigenvalues: {metrics['eigenvalues']}")


if __name__ == "__main__":
    example_usage()
