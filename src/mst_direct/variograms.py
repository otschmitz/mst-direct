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


def _lag_stratified_grid_pairs(
    coords: np.ndarray,
    edges: np.ndarray,
    n_pairs: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Sample an equal number of regular-grid pairs in every lag bin.

    Offsets are sampled in proportion to the number of pairs they represent,
    then a valid origin is sampled uniformly.  Thus each bin estimates the
    usual all-pairs variogram without starving short lags.  Returns ``None``
    when coordinates are not a complete regular Cartesian grid.
    """
    xs = np.unique(coords[:, 0])
    ys = np.unique(coords[:, 1])
    nx, ny = len(xs), len(ys)
    if nx * ny != len(coords):
        return None
    if nx > 2 and not np.allclose(np.diff(xs), np.diff(xs)[0]):
        return None
    if ny > 2 and not np.allclose(np.diff(ys), np.diff(ys)[0]):
        return None
    xi = np.searchsorted(xs, coords[:, 0])
    yi = np.searchsorted(ys, coords[:, 1])
    grid = np.full((ny, nx), -1, dtype=np.int64)
    grid[yi, xi] = np.arange(len(coords))
    if np.any(grid < 0):
        return None

    sx = float(np.diff(xs)[0]) if nx > 1 else 1.0
    sy = float(np.diff(ys)[0]) if ny > 1 else 1.0
    offsets: list[list[tuple[int, int, int]]] = [[] for _ in range(len(edges) - 1)]
    for dy in range(0, ny):
        for dx in range(-(nx - 1), nx):
            if dy == 0 and dx <= 0:
                continue
            dist = np.hypot(dx * sx, dy * sy)
            b = int(np.searchsorted(edges, dist, side="right") - 1)
            if 0 <= b < len(offsets) and dist <= edges[-1]:
                weight = (nx - abs(dx)) * (ny - dy)
                offsets[b].append((dx, dy, weight))

    per_bin = max(1, int(np.ceil(n_pairs / max(len(offsets), 1))))
    all_i: list[np.ndarray] = []
    all_j: list[np.ndarray] = []
    for choices in offsets:
        if not choices:
            continue
        arr = np.asarray(choices, dtype=np.int64)
        probabilities = arr[:, 2] / arr[:, 2].sum()
        selected = rng.choice(len(arr), size=per_bin, p=probabilities)
        dx = arr[selected, 0]
        dy = arr[selected, 1]
        xlo = np.maximum(0, -dx)
        xhi = np.minimum(nx, nx - dx)
        ox = xlo + (rng.random(per_bin) * (xhi - xlo)).astype(np.int64)
        oy = (rng.random(per_bin) * (ny - dy)).astype(np.int64)
        all_i.append(grid[oy, ox])
        all_j.append(grid[oy + dy, ox + dx])
    return np.concatenate(all_i), np.concatenate(all_j)


def sampled_variogram(
    values: np.ndarray,
    coords: np.ndarray,
    n_lags: int = 20,
    max_dist: float | None = None,
    n_pairs: int = 400_000,
    random_state: int | None = 0,
    sampling: str = "lag_stratified",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Omnidirectional experimental variogram from sampled pairs.

    Returns ``(lags, gamma, counts)`` like :func:`empirical_variogram`, but
    ``O(n_pairs)`` in time and memory instead of ``O(n^2)``.
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    coords = np.asarray(coords, dtype=np.float64)
    n = len(values)
    if max_dist is None:
        span = np.ptp(coords, axis=0)
        max_dist = float(np.hypot(span[0], span[1]) * 0.6)
    edges = np.linspace(0.0, max_dist, n_lags + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    rng = np.random.default_rng(random_state)
    if sampling == "lag_stratified":
        pairs = _lag_stratified_grid_pairs(coords, edges, n_pairs, rng)
        if pairs is None:
            raise ValueError("lag_stratified sampling requires a complete regular grid")
        i, j = pairs
    elif sampling == "uniform":
        i, j = _pair_sample(n, n_pairs, rng)
    else:
        raise ValueError("sampling must be 'lag_stratified' or 'uniform'")

    d = np.hypot(coords[i, 0] - coords[j, 0], coords[i, 1] - coords[j, 1])
    sq = 0.5 * (values[i] - values[j]) ** 2
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
    sampling: str = "lag_stratified",
) -> tuple[np.ndarray, np.ndarray]:
    """Experimental cross-variogram ``0.5 E[(a_i-a_j)(b_i-b_j)]`` from pairs."""
    va = np.asarray(va, dtype=np.float64).ravel()
    vb = np.asarray(vb, dtype=np.float64).ravel()
    coords = np.asarray(coords, dtype=np.float64)
    n = len(va)
    if max_dist is None:
        span = np.ptp(coords, axis=0)
        max_dist = float(np.hypot(span[0], span[1]) * 0.6)
    edges = np.linspace(0.0, max_dist, n_lags + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    rng = np.random.default_rng(random_state)
    if sampling == "lag_stratified":
        pairs = _lag_stratified_grid_pairs(coords, edges, n_pairs, rng)
        if pairs is None:
            raise ValueError("lag_stratified sampling requires a complete regular grid")
        i, j = pairs
    elif sampling == "uniform":
        i, j = _pair_sample(n, n_pairs, rng)
    else:
        raise ValueError("sampling must be 'lag_stratified' or 'uniform'")
    d = np.hypot(coords[i, 0] - coords[j, 0], coords[i, 1] - coords[j, 1])
    cross = 0.5 * (va[i] - va[j]) * (vb[i] - vb[j])
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
