"""
Phronesis Index — Spectral Sheaf Heuristics for Consistency Detection
in Multi-Agent Systems.

This package provides a computationally efficient spectral heuristic
(the Phronesis Index, Φ) that quantifies global consistency of a
multi-agent system by approximating topological obstructions
(cohomology) in cellular sheaves.

Reference
---------
Bayat, S. (2026). Spectral Sheaf Heuristics for Consistency Detection
in Multi-Agent Systems. *Nature Machine Intelligence* (submitted).
"""

from phronesis.core import PhronesisIndex
from phronesis.laplacian import build_connection_laplacian
from phronesis.epsilon import select_epsilon, adaptive_epsilon

__version__ = "1.0.0"
__all__ = [
    "PhronesisIndex",
    "build_connection_laplacian",
    "select_epsilon",
    "adaptive_epsilon",
]
