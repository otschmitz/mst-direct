"""Comparison methods: Gaussian Copula and LU Decomposition (Section IV-C).

These are the two traditional multivariate simulation methods the paper
compares MST-Direct against. Both assume an underlying *Gaussian* dependence
structure, which is exactly why they fail to reproduce complex non-linear joint
distributions (producing the elliptical scatter patterns of Fig. 2) while
MST-Direct preserves them.

They require the optional ``[baselines]`` extra::

    pip install 'mst-direct[baselines]'
"""

from __future__ import annotations

import numpy as np

from .adjacency import pairwise_distances
from .variograms import exponential

__all__ = ["gaussian_copula_simulate", "lu_simulate"]


def _require_scipy():
    try:
        import scipy.stats  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "scipy is required for the baseline simulators. "
            "Install it with: pip install 'mst-direct[baselines]'"
        ) from exc
    import scipy.stats as st

    return st


def _normal_scores(values: np.ndarray, st) -> np.ndarray:
    """Map each column to standard-normal scores via its empirical rank."""
    values = np.asarray(values, dtype=np.float64)
    n = values.shape[0]
    scores = np.empty_like(values)
    for c in range(values.shape[1]):
        ranks = st.rankdata(values[:, c], method="average")
        scores[:, c] = st.norm.ppf(ranks / (n + 1.0))
    return scores


def _back_transform(gaussian: np.ndarray, marginal: np.ndarray) -> np.ndarray:
    """Rank-match a Gaussian field to a target empirical marginal.

    The ``i``-th largest simulated value is assigned the ``i``-th largest
    original value, so the output reproduces the original marginal exactly.
    """
    order = np.argsort(np.argsort(gaussian))
    sorted_marginal = np.sort(marginal)
    return sorted_marginal[order]


def _spatial_correlation(coords: np.ndarray, rng_range: float) -> np.ndarray:
    """Exponential spatial correlation matrix ``C = 1 - gamma`` over coords."""
    dist = pairwise_distances(coords)
    return 1.0 - exponential(dist, sill=1.0, rng=rng_range)


def _default_range(coords: np.ndarray) -> float:
    extent = float(np.ptp(coords, axis=0).max())
    return max(extent * 0.25, 1.0)


def _chol(matrix: np.ndarray, jitter: float = 1e-8) -> np.ndarray:
    """Cholesky factor with incremental jitter for numerical PSD safety."""
    a = np.asarray(matrix, dtype=np.float64)
    eye = np.eye(a.shape[0])
    for scale in (0.0, jitter, jitter * 100, jitter * 1e4, jitter * 1e6):
        try:
            return np.linalg.cholesky(a + scale * eye)
        except np.linalg.LinAlgError:
            continue
    raise np.linalg.LinAlgError("covariance is not positive definite even with jitter")


def gaussian_copula_simulate(
    values: np.ndarray,
    coords: np.ndarray,
    spatial_range: float | None = None,
    random_state: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Gaussian-copula spatial simulation (paper's Copula baseline).

    Transforms each variable to Gaussian marginals (normal scores), estimates
    the inter-variable correlation in Gaussian space, simulates spatially
    correlated Gaussian fields carrying that cross-correlation, and
    back-transforms to the original marginals. The joint dependence is forced
    to be Gaussian/elliptical.

    Args:
        values: ``(n, d)`` original data.
        coords: ``(n, 2)`` coordinates.
        spatial_range: Exponential correlation range (default: 25% of extent).
        random_state: Seed or generator.

    Returns:
        ``(n, d)`` simulated field with the original marginals preserved.
    """
    st = _require_scipy()
    values = np.asarray(values, dtype=np.float64)
    coords = np.asarray(coords, dtype=np.float64)
    n, d = values.shape
    rng = np.random.default_rng(random_state)
    if spatial_range is None:
        spatial_range = _default_range(coords)

    scores = _normal_scores(values, st)
    cross = np.corrcoef(scores, rowvar=False)
    cross = np.atleast_2d(cross)

    spatial = _spatial_correlation(coords, spatial_range)
    l_spatial = _chol(spatial)
    l_cross = _chol(cross)

    # Independent spatially-correlated standard fields, then mix across vars.
    white = rng.standard_normal((n, d))
    spatial_fields = l_spatial @ white  # (n, d), each column spatially correlated
    correlated = spatial_fields @ l_cross.T  # impose inter-variable correlation

    out = np.empty_like(values)
    for c in range(d):
        out[:, c] = _back_transform(correlated[:, c], values[:, c])
    return out


def lu_simulate(
    values: np.ndarray,
    coords: np.ndarray,
    spatial_range: float | None = None,
    random_state: int | np.random.Generator | None = None,
) -> np.ndarray:
    """LU-decomposition spatial simulation (paper's LU baseline).

    Builds the joint covariance as the Kronecker product of the inter-variable
    correlation matrix and the spatial covariance, factorizes it (Cholesky /
    LU), simulates a jointly Gaussian field, and maps each variable back to its
    original marginal. Produces inherently Gaussian (elliptical) joints.

    Args:
        values: ``(n, d)`` original data.
        coords: ``(n, 2)`` coordinates.
        spatial_range: Exponential covariance range (default: 25% of extent).
        random_state: Seed or generator.

    Returns:
        ``(n, d)`` simulated field with the original marginals preserved.
    """
    _require_scipy()
    values = np.asarray(values, dtype=np.float64)
    coords = np.asarray(coords, dtype=np.float64)
    n, d = values.shape
    rng = np.random.default_rng(random_state)
    if spatial_range is None:
        spatial_range = _default_range(coords)

    cross = np.atleast_2d(np.corrcoef(values, rowvar=False))
    spatial = _spatial_correlation(coords, spatial_range)

    # Joint covariance = cross-correlation (Kronecker) average spatial covariance.
    joint = np.kron(cross, spatial)  # (d*n, d*n)
    factor = _chol(joint)

    z = factor @ rng.standard_normal(d * n)
    fields = z.reshape(d, n).T  # (n, d)

    out = np.empty_like(values)
    for c in range(d):
        out[:, c] = _back_transform(fields[:, c], values[:, c])
    return out
