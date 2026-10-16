"""Variogram models and empirical variogram estimation.

Provides the theoretical models used in the paper's experiments — the
**spherical** model (eq. 12) for variable X and the **exponential** model
(eq. 13) for variable Y — an empirical (experimental) variogram estimator, and
the variogram-preservation metric (eq. 15).
"""

from __future__ import annotations

import numpy as np

from .adjacency import pairwise_distances

__all__ = [
    "spherical",
    "exponential",
    "gaussian",
    "empirical_variogram",
    "variogram_correlation",
    # scalable estimators (v2)
    "grid_coords",
    "sampled_variogram",
    "sampled_cross_variogram",
    "fit_spherical",
]


def spherical(
    h: np.ndarray, sill: float = 1.0, rng: float = 1.0, nugget: float = 0.0
) -> np.ndarray:
    """Spherical variogram model (eq. 12).

    ``gamma(h) = nugget + (sill - nugget) * [3h/2a - 1/2 (h/a)^3]`` for
    ``h <= a`` (the range) and ``sill`` beyond it.
    """
    h = np.asarray(h, dtype=np.float64)
    a = float(rng)
    ratio = np.clip(h / a, 0.0, 1.0)
    structured = 1.5 * ratio - 0.5 * ratio**3
    out = nugget + (sill - nugget) * structured
    return np.where(h <= 0, 0.0, out)


def exponential(
    h: np.ndarray, sill: float = 1.0, rng: float = 1.0, nugget: float = 0.0
) -> np.ndarray:
    """Exponential variogram model (eq. 13).

    ``gamma(h) = nugget + (sill - nugget) * [1 - exp(-3h/a)]``. The factor 3
    makes ``a`` the *practical* range (95% of the sill).
    """
    h = np.asarray(h, dtype=np.float64)
    a = float(rng)
    out = nugget + (sill - nugget) * (1.0 - np.exp(-3.0 * h / a))
    return np.where(h <= 0, 0.0, out)


def gaussian(h: np.ndarray, sill: float = 1.0, rng: float = 1.0, nugget: float = 0.0) -> np.ndarray:
    """Gaussian variogram model: ``nugget + (sill-nugget)*[1-exp(-3(h/a)^2)]``."""
    h = np.asarray(h, dtype=np.float64)
    a = float(rng)
    out = nugget + (sill - nugget) * (1.0 - np.exp(-3.0 * (h / a) ** 2))
    return np.where(h <= 0, 0.0, out)


def empirical_variogram(
    values: np.ndarray,
    coords: np.ndarray,
    n_lags: int = 15,
    max_dist: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate the experimental (omnidirectional) variogram.

    ``gamma(h) = 1 / (2 N(h)) * sum_{(i,j) in h} (z_i - z_j)^2`` over all point
    pairs whose separation falls in each distance bin.

    Args:
        values: ``(n,)`` variable values at each location.
        coords: ``(n, 2)`` coordinates.
        n_lags: Number of distance bins.
        max_dist: Maximum lag distance (default: the maximum pairwise distance).

    Returns:
        ``(lags, gamma, counts)`` — bin centers, semivariance per bin, and the
        number of pairs contributing to each bin. Empty bins yield ``NaN``.
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    coords = np.asarray(coords, dtype=np.float64)
    dist = pairwise_distances(coords)

    iu, ju = np.triu_indices(len(values), k=1)
    d = dist[iu, ju]
    sq_diff = (values[iu] - values[ju]) ** 2

    if max_dist is None:
        max_dist = float(d.max())
    edges = np.linspace(0.0, max_dist, n_lags + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    gamma = np.full(n_lags, np.nan)
    counts = np.zeros(n_lags, dtype=np.int64)
    bin_idx = np.digitize(d, edges) - 1
    for b in range(n_lags):
        mask = bin_idx == b
        c = int(mask.sum())
        counts[b] = c
        if c > 0:
            gamma[b] = 0.5 * sq_diff[mask].mean()
    return centers, gamma, counts


def variogram_correlation(gamma_a: np.ndarray, gamma_b: np.ndarray) -> float:
    """Pearson correlation between two variograms (eq. 15).

    NaN bins (empty lags) present in either input are dropped pairwise.
    """
    a = np.asarray(gamma_a, dtype=np.float64)
    b = np.asarray(gamma_b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return float("nan")
    a, b = a[mask], b[mask]
    if np.std(a) < 1e-15 or np.std(b) < 1e-15:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


# ---------------------------------------------------------------------------
# Scalable estimators (v2): omnidirectional experimental (cross-)variograms
# from sampled point pairs, O(n_pairs) instead of the O(n^2) full estimator
# above, plus a spherical-model fit. Used by the multivariate/large-grid
# workflows where the full pairwise matrix is infeasible.
# ---------------------------------------------------------------------------


def grid_coords(nx: int, ny: int | None = None) -> np.ndarray:
    """Row-major ``(nx*ny, 2)`` integer coordinates of a regular grid.

    Uses the same ravel order as ``reshape(value, (ny, nx))`` with ``(x, y)``
    columns, matching :func:`mst_direct.synthetic.make_grid`.
    """
    ny = nx if ny is None else ny
    ys, xs = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    return np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float64)


def _pair_sample(
    n: int, n_pairs: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    keep = i != j
    return i[keep], j[keep]


def sampled_variogram(
    values: np.ndarray,
    coords: np.ndarray,
    n_lags: int = 20,
    max_dist: float | None = None,
    n_pairs: int = 400_000,
    random_state: int | None = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Omnidirectional experimental variogram from sampled pairs.

    Returns ``(lags, gamma, counts)`` like :func:`empirical_variogram`, but
    ``O(n_pairs)`` in time and memory instead of ``O(n^2)``.
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    coords = np.asarray(coords, dtype=np.float64)
    n = len(values)
    rng = np.random.default_rng(random_state)
    i, j = _pair_sample(n, n_pairs, rng)

    d = np.hypot(coords[i, 0] - coords[j, 0], coords[i, 1] - coords[j, 1])
    sq = 0.5 * (values[i] - values[j]) ** 2

    if max_dist is None:
        max_dist = float(np.percentile(d, 60))
    edges = np.linspace(0.0, max_dist, n_lags + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.digitize(d, edges) - 1, 0, n_lags - 1)
    inside = d <= max_dist

    gamma = np.full(n_lags, np.nan)
    counts = np.zeros(n_lags, dtype=np.int64)
    for b in range(n_lags):
        mask = inside & (idx == b)
        c = int(mask.sum())
        counts[b] = c
        if c > 0:
            gamma[b] = sq[mask].mean()
    return centers, gamma, counts


def sampled_cross_variogram(
    va: np.ndarray,
    vb: np.ndarray,
    coords: np.ndarray,
    n_lags: int = 20,
    max_dist: float | None = None,
    n_pairs: int = 400_000,
    random_state: int | None = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Experimental cross-variogram ``0.5 E[(a_i-a_j)(b_i-b_j)]`` from pairs."""
    va = np.asarray(va, dtype=np.float64).ravel()
    vb = np.asarray(vb, dtype=np.float64).ravel()
    coords = np.asarray(coords, dtype=np.float64)
    n = len(va)
    rng = np.random.default_rng(random_state)
    i, j = _pair_sample(n, n_pairs, rng)
    d = np.hypot(coords[i, 0] - coords[j, 0], coords[i, 1] - coords[j, 1])
    cross = 0.5 * (va[i] - va[j]) * (vb[i] - vb[j])
    if max_dist is None:
        max_dist = float(np.percentile(d, 60))
    edges = np.linspace(0.0, max_dist, n_lags + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.digitize(d, edges) - 1, 0, n_lags - 1)
    inside = d <= max_dist
    gamma = np.full(n_lags, np.nan)
    for b in range(n_lags):
        mask = inside & (idx == b)
        if mask.any():
            gamma[b] = cross[mask].mean()
    return centers, gamma


def fit_spherical(
    lags: np.ndarray,
    gamma: np.ndarray,
    sill: float | None = None,
) -> tuple[float, float]:
    """Least-squares fit of a spherical model to an experimental variogram.

    Returns ``(range, sill)``. The sill defaults to the experimental variance
    estimate (mean of the outer half of the lags) if not supplied.
    """
    lags = np.asarray(lags, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    mask = np.isfinite(gamma)
    lags, gamma = lags[mask], gamma[mask]
    if sill is None:
        sill = float(np.nanmean(gamma[len(gamma) // 2:])) if len(gamma) else 1.0

    candidates = np.linspace(max(lags.min(), 1e-3), lags.max() * 1.5, 200)
    best_rng, best_err = candidates[0], np.inf
    for a in candidates:
        model = spherical(lags, sill=sill, rng=a)
        err = float(np.mean((model - gamma) ** 2))
        if err < best_err:
            best_err, best_rng = err, a
    return float(best_rng), float(sill)
