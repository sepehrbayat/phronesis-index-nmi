"""
Connection Laplacian construction for cellular sheaves.

The Connection Laplacian L is the block matrix L = B^T B where B is
the coboundary operator of the cellular sheaf.

Block structure (size Nd × Nd):

    L[v, v] = Σ_{e ∋ v} r_{e,v}^T r_{e,v}     (diagonal blocks)
    L[u, v] = −r_{e,u}^T r_{e,v}                (off-diagonal, e = (u,v))
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix


def build_connection_laplacian(
    graph: nx.Graph,
    stalks: Dict[int, int],
    restriction_maps: Dict[Tuple[int, int], np.ndarray],
) -> Tuple[csr_matrix, int]:
    """Construct the Connection Laplacian L = B^T B as a sparse matrix.

    This uses the coboundary-operator factorisation:

    1. Build boundary operator B  (maps vertex sections → edge sections).
    2. Return L = B^T B.

    Parameters
    ----------
    graph : nx.Graph
        Communication graph. Vertices = agents, edges = links.
    stalks : dict
        ``{vertex_id: stalk_dimension}``.  Each stalk is ℝ^d.
    restriction_maps : dict
        ``{(u, v): R_uv}`` where ``R_uv`` is a ``(d_v, d_u)`` ndarray.
        Missing edges default to the identity matrix.

    Returns
    -------
    L : scipy.sparse.csr_matrix
        Connection Laplacian of shape ``(total_dim, total_dim)``.
    total_dim : int
        Sum of all stalk dimensions.
    """
    nodes = list(graph.nodes())
    vertex_to_idx = {v: i for i, v in enumerate(nodes)}

    # Pre-compute cumulative stalk offsets
    offsets: Dict[int, int] = {}
    running = 0
    for v in nodes:
        offsets[v] = running
        running += stalks[v]
    total_dim = running

    # ---- Build coboundary matrix B ----
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []

    edge_offset = 0
    for u, v in graph.edges():
        d_u = stalks[u]
        d_v = stalks[v]
        R_uv = restriction_maps.get((u, v), np.eye(d_v, d_u))

        u_start = offsets[u]
        v_start = offsets[v]

        # Entries for R_uv (from u) and −I (from v)
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

    B = csr_matrix((data, (rows, cols)), shape=(edge_offset, total_dim))
    L = B.T @ B
    return L, total_dim
