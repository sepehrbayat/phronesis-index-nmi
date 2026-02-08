"""Unit tests for the phronesis package."""

import numpy as np
import networkx as nx
import pytest

from phronesis import PhronesisIndex, build_connection_laplacian, select_epsilon


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _triangle_sheaf():
    """3-node cycle with identity restriction maps (fully consistent)."""
    G = nx.cycle_graph(3)
    stalks = {0: 2, 1: 2, 2: 2}
    maps = {(0, 1): np.eye(2), (1, 2): np.eye(2)}
    return G, stalks, maps


def _line_sheaf():
    """2-node path — simplest non-trivial sheaf."""
    G = nx.path_graph(2)
    stalks = {0: 2, 1: 2}
    maps = {(0, 1): np.eye(2)}
    return G, stalks, maps


def _inconsistent_triangle():
    """Triangle with one perturbed restriction map (inconsistent)."""
    G = nx.cycle_graph(3)
    stalks = {0: 2, 1: 2, 2: 2}
    maps = {
        (0, 1): np.eye(2),
        (1, 2): np.eye(2) + np.array([[0.0, 0.8], [-0.8, 0.0]]),
    }
    return G, stalks, maps


# ---------------------------------------------------------------------------
# Laplacian tests
# ---------------------------------------------------------------------------

class TestLaplacian:
    def test_shape(self):
        G, stalks, maps = _triangle_sheaf()
        L, dim = build_connection_laplacian(G, stalks, maps)
        assert L.shape == (dim, dim)
        assert dim == 6  # 3 nodes × 2

    def test_symmetric(self):
        G, stalks, maps = _triangle_sheaf()
        L, _ = build_connection_laplacian(G, stalks, maps)
        diff = L - L.T
        assert diff.nnz == 0 or np.max(np.abs(diff.toarray())) < 1e-12

    def test_positive_semidefinite(self):
        G, stalks, maps = _triangle_sheaf()
        L, _ = build_connection_laplacian(G, stalks, maps)
        eigenvalues = np.linalg.eigvalsh(L.toarray())
        assert np.all(eigenvalues >= -1e-10)


# ---------------------------------------------------------------------------
# PhronesisIndex tests
# ---------------------------------------------------------------------------

class TestPhronesisIndex:
    def test_consistent_system_high_phi(self):
        G, stalks, maps = _triangle_sheaf()
        idx = PhronesisIndex(G, stalks, maps)
        result = idx.compute(epsilon=0.01)
        assert result.phi > 0
        assert result.h1_epsilon == 0  # no topological holes

    def test_inconsistent_system_detectable(self):
        G, stalks, maps = _inconsistent_triangle()
        idx = PhronesisIndex(G, stalks, maps)
        result = idx.compute(epsilon=0.01)
        # Phi should still be computable
        assert result.phi > 0

    def test_line_graph(self):
        G, stalks, maps = _line_sheaf()
        idx = PhronesisIndex(G, stalks, maps)
        result = idx.compute(epsilon=0.01, k=3)
        assert result.phi > 0

    def test_result_fields(self):
        G, stalks, maps = _triangle_sheaf()
        idx = PhronesisIndex(G, stalks, maps)
        result = idx.compute(epsilon=0.05)
        assert hasattr(result, "phi")
        assert hasattr(result, "lambda_1_plus")
        assert hasattr(result, "h1_epsilon")
        assert hasattr(result, "epsilon")
        assert hasattr(result, "eigenvalues")
        assert result.epsilon == 0.05

    def test_legacy_get_metrics(self):
        G, stalks, maps = _triangle_sheaf()
        idx = PhronesisIndex(G, stalks, maps)
        m = idx.get_metrics()
        assert "phi" in m and "eigenvalues" in m

    def test_deterministic_with_seed(self):
        """Same input → same output (no randomness in Lanczos for small k)."""
        G, stalks, maps = _triangle_sheaf()
        r1 = PhronesisIndex(G, stalks, maps).compute(epsilon=0.01)
        r2 = PhronesisIndex(G, stalks, maps).compute(epsilon=0.01)
        assert abs(r1.phi - r2.phi) < 1e-10


# ---------------------------------------------------------------------------
# Epsilon selection tests
# ---------------------------------------------------------------------------

class TestEpsilon:
    def test_select_epsilon_basic(self):
        eigs = np.array([0.0, 0.0, 0.001, 1.5, 2.0, 3.0])
        eps = select_epsilon(eigs)
        assert 0 < eps < 1.5

    def test_select_epsilon_single(self):
        eps = select_epsilon(np.array([1.0]))
        assert eps > 0


# ---------------------------------------------------------------------------
# Smoke test for larger graph
# ---------------------------------------------------------------------------

class TestScalability:
    def test_grid_graph_runs(self):
        """Ensure a 10×10 grid completes without error."""
        G = nx.grid_2d_graph(10, 10)
        mapping = {n: i for i, n in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, mapping)
        stalks = {v: 2 for v in G.nodes()}
        maps = {(u, v): np.eye(2) for u, v in G.edges()}
        idx = PhronesisIndex(G, stalks, maps)
        result = idx.compute(epsilon=0.01, k=20)
        assert result.phi > 0
        assert result.eigenvalues.shape[0] == 20
