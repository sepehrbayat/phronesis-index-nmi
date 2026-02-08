"""
Epsilon (ε) selection procedures for the Phronesis Index.

Two strategies are provided:
1. ``select_epsilon`` — spectral-gap heuristic (default for most uses).
2. ``adaptive_epsilon`` — noise-adaptive procedure from Sec 3.4 of the paper.
"""

from __future__ import annotations

import numpy as np


def select_epsilon(
    eigenvalues: np.ndarray,
    *,
    factor: float = 0.5,
    min_epsilon: float = 1e-6,
) -> float:
    """Choose ε as half the spectral gap.

    This implements the default procedure recommended in the paper
    (Definition 1): ε = δ/2 where δ is the gap between the largest
    near-zero eigenvalue cluster and the first clearly positive one.

    Parameters
    ----------
    eigenvalues : ndarray
        Sorted (ascending) eigenvalues of the Connection Laplacian.
    factor : float
        Fraction of the detected gap to use (default 0.5 = δ/2).
    min_epsilon : float
        Floor value to avoid numerical zeros.

    Returns
    -------
    epsilon : float
    """
    eigs = np.sort(np.real(eigenvalues))

    if len(eigs) < 2:
        return min_epsilon

    # Compute successive gaps
    gaps = np.diff(eigs)
    if len(gaps) == 0:
        return min_epsilon

    # Find the largest gap (spectral gap δ)
    idx = int(np.argmax(gaps))
    delta = gaps[idx]

    epsilon = max(factor * delta, min_epsilon)
    return float(epsilon)


def adaptive_epsilon(
    eigenvalues: np.ndarray,
    *,
    noise_std: float | None = None,
    spectral_gap: float | None = None,
    safety_factor: float = 4.0,
) -> float:
    """Noise-adaptive ε selection (Sec. 3.4).

    If noise level σ and spectral gap δ are known, sets
    ε = δ/2 and validates that σ < δ / safety_factor.

    Parameters
    ----------
    eigenvalues : ndarray
        Sorted eigenvalues.
    noise_std : float or None
        Estimated noise level σ.  If None, uses default heuristic.
    spectral_gap : float or None
        Known spectral gap δ.  If None, estimated from eigenvalues.
    safety_factor : float
        Require σ < δ / safety_factor for the bound to hold (default 4).

    Returns
    -------
    epsilon : float
    """
    eigs = np.sort(np.real(eigenvalues))

    if spectral_gap is None:
        gaps = np.diff(eigs)
        spectral_gap = float(np.max(gaps)) if len(gaps) > 0 else 0.01

    if noise_std is not None and noise_std >= spectral_gap / safety_factor:
        import warnings
        warnings.warn(
            f"Noise σ={noise_std:.4f} ≥ δ/{safety_factor}={spectral_gap/safety_factor:.4f}. "
            "Error bound may not hold (Theorem 2).",
            stacklevel=2,
        )

    return max(spectral_gap / 2.0, 1e-6)
