"""
Core Phronesis Index computation.

Implements Definition 1 from the paper:

    Φ = λ₁⁺ / (h¹_ε + ε)

where
    λ₁⁺  = smallest eigenvalue ≥ ε  (spectral gap indicator),
    h¹_ε = #{i : λ_i < ε} − 1      (approximate 1st cohomology dim),
    ε    = threshold (see ``phronesis.epsilon``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigsh

from phronesis.laplacian import build_connection_laplacian


@dataclass
class PhronesisResult:
    """Container for a single Φ computation."""

    phi: float
    lambda_1_plus: float
    h1_epsilon: int
    epsilon: float
    eigenvalues: np.ndarray = field(repr=False)


class PhronesisIndex:
    """Compute the Phronesis Index for a cellular sheaf on a graph.

    Parameters
    ----------
    graph : nx.Graph
        Communication graph (vertices = agents, edges = links).
    stalks : dict
        ``{vertex_id: stalk_dim}``.
    restriction_maps : dict
        ``{(u, v): R_uv}`` — restriction map matrices.

    Examples
    --------
    >>> import networkx as nx, numpy as np
    >>> G = nx.cycle_graph(3)
    >>> stalks = {0: 2, 1: 2, 2: 2}
    >>> maps = {(0,1): np.eye(2), (1,2): np.eye(2), (2,0): np.eye(2)}
    >>> idx = PhronesisIndex(G, stalks, maps)
    >>> result = idx.compute(epsilon=0.01)
    >>> result.phi > 0
    True
    """

    def __init__(
        self,
        graph: nx.Graph,
        stalks: Dict[int, int],
        restriction_maps: Dict[Tuple[int, int], np.ndarray],
    ) -> None:
        self.graph = graph
        self.stalks = stalks
        self.restriction_maps = restriction_maps
        self._laplacian, self._total_dim = build_connection_laplacian(
            graph, stalks, restriction_maps
        )

    # ------------------------------------------------------------------
    def compute(self, epsilon: float = 0.01, k: int = 20) -> PhronesisResult:
        """Compute the Phronesis Index Φ.

        Parameters
        ----------
        epsilon : float
            Threshold for near-zero eigenvalue classification.
        k : int
            Number of smallest eigenvalues to compute via Lanczos.

        Returns
        -------
        PhronesisResult
        """
        k_eff = min(k, self._total_dim - 1)
        if k_eff < 1:
            return PhronesisResult(
                phi=0.0,
                lambda_1_plus=0.0,
                h1_epsilon=0,
                epsilon=epsilon,
                eigenvalues=np.array([0.0]),
            )

        eigenvalues = eigsh(
            self._laplacian, k=k_eff, which="SM", return_eigenvectors=False
        )
        eigenvalues = np.sort(np.real(eigenvalues))

        # λ₁⁺ — smallest eigenvalue ≥ ε
        positive_eigs = eigenvalues[eigenvalues >= epsilon]
        lambda_1_plus = float(positive_eigs[0]) if len(positive_eigs) > 0 else epsilon

        # h¹_ε — count near-zero eigenvalues, subtract 1 for H⁰
        h1_epsilon = int(np.sum(eigenvalues < epsilon)) - 1
        h1_epsilon = max(h1_epsilon, 0)

        phi = lambda_1_plus / (h1_epsilon + epsilon)

        return PhronesisResult(
            phi=phi,
            lambda_1_plus=lambda_1_plus,
            h1_epsilon=h1_epsilon,
            epsilon=epsilon,
            eigenvalues=eigenvalues,
        )

    # ------------------------------------------------------------------
    # Convenience aliases
    # ------------------------------------------------------------------
    def get_metrics(self) -> dict:
        """Return last-computed metrics as a plain dict (legacy API)."""
        r = self.compute()
        return {
            "phi": r.phi,
            "lambda_1_plus": r.lambda_1_plus,
            "h1_epsilon": r.h1_epsilon,
            "epsilon": r.epsilon,
            "eigenvalues": r.eigenvalues,
        }
