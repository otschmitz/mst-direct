"""Projection Pursuit Multivariate Transform (Barnett, Manchuk & Deutsch, 2014).

PPMT is the comparator used throughout DMS Section 4. It iteratively
Gaussianizes a multivariate sample: sphere the data (PCA whitening), then
repeatedly find the most non-Gaussian 1-D projection and normal-score transform
along it, until the sample is approximately multivariate Gaussian with no cross
correlation. The transform is invertible, so geostatistical simulation is:

1. ``PPMT`` the reference cloud  ->  independent Gaussian factors;
2. simulate independent spatially-correlated Gaussian fields (FFT-MA) with the
   prescribed variogram (optionally conditioned on Gaussianized hard data);
3. apply the inverse PPMT  ->  a realization following the target distribution.

This is a faithful, compact reimplementation: the projection index is the
non-Gaussianity score ``skew^2 + (kurtosis - 3)^2`` maximized over random
directions, and each projection is Gaussianized with an empirical
normal-score transform (linear interpolation + linear tail extrapolation).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

__all__ = ["PPMT", "ppmt_simulate"]


def _nscore_forward(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Empirical normal-score of ``x``; return ``(gauss, sorted_x, sorted_g)``."""
    n = len(x)
    order = np.argsort(x, kind="stable")
    ranks = np.empty(n)
    # average ranks for ties via probability points (i + 0.5)/n on sorted order
    pp = (np.arange(n) + 0.5) / n
    g_sorted = norm.ppf(pp)
    x_sorted = x[order]
    gauss = np.empty(n)
    gauss[order] = g_sorted
    return gauss, x_sorted, g_sorted


def _interp_extrap(
    query: np.ndarray, xp: np.ndarray, fp: np.ndarray, clip: bool = False
) -> np.ndarray:
    """Linear interpolation beyond the table ends.

    With ``clip=True`` the value is held at the table endpoints instead of
    being linearly extrapolated. This is used on the *inverse* (Gaussian ->
    data) transform: simulated Gaussian factors reach further into the tails
    than the fitted data, and unbounded extrapolation of a heavy-tailed
    marginal would inflate the variance (and the variogram sill).
    """
    out = np.interp(query, xp, fp)
    if clip or len(xp) <= 1:
        return out
    lo = query < xp[0]
    if lo.any():
        slope = (fp[1] - fp[0]) / (xp[1] - xp[0] + 1e-12)
        out[lo] = fp[0] + slope * (query[lo] - xp[0])
    hi = query > xp[-1]
    if hi.any():
        slope = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2] + 1e-12)
        out[hi] = fp[-1] + slope * (query[hi] - xp[-1])
    return out


def _projection_index(p: np.ndarray) -> float:
    """Non-Gaussianity of a 1-D sample: ``skew^2 + (kurtosis - 3)^2``."""
    p = p - p.mean()
    s = p.std()
    if s < 1e-12:
        return 0.0
    z = p / s
    skew = np.mean(z**3)
    kurt = np.mean(z**4)
    return float(skew**2 + (kurt - 3.0) ** 2)


class PPMT:
    """Forward/inverse Projection Pursuit Multivariate Transform.

    Args:
        n_iter: Number of projection-pursuit iterations.
        n_dirs: Random unit directions tried per iteration (the best by the
            projection index is selected).
        random_state: Seed for the random directions.
    """

    def __init__(
        self, n_iter: int = 30, n_dirs: int = 60, random_state: int | None = None
    ) -> None:
        self.n_iter = int(n_iter)
        self.n_dirs = int(n_dirs)
        self.random_state = random_state
        self.mean_: np.ndarray | None = None
        self.whiten_: np.ndarray | None = None
        self.dewhiten_: np.ndarray | None = None
        self.steps_: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self.factors_: np.ndarray | None = None  # forward factors of the training data

    def fit(self, x: np.ndarray) -> "PPMT":
        """Fit the transform on the multivariate sample ``x`` ``(n, d)``."""
        x = np.asarray(x, dtype=np.float64)
        rng = np.random.default_rng(self.random_state)
        self.mean_ = x.mean(axis=0)
        xc = x - self.mean_
        cov = np.cov(xc, rowvar=False)
        evals, evecs = np.linalg.eigh(cov)
        evals = np.clip(evals, 1e-10, None)
        self.whiten_ = evecs @ np.diag(evals**-0.5) @ evecs.T
        self.dewhiten_ = evecs @ np.diag(evals**0.5) @ evecs.T
        w = xc @ self.whiten_

        d = x.shape[1]
        self.steps_ = []
        for _ in range(self.n_iter):
            dirs = rng.standard_normal((self.n_dirs, d))
            dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
            scores = [_projection_index(w @ p) for p in dirs]
            p = dirs[int(np.argmax(scores))]
            proj = w @ p
            gauss, x_sorted, g_sorted = _nscore_forward(proj)
            self.steps_.append((p, x_sorted, g_sorted))
            w = w + np.outer(gauss - proj, p)
        self.factors_ = w  # final Gaussian factors of the training data
        return self

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Transform ``x`` to the (approximately) Gaussian factor space."""
        x = np.asarray(x, dtype=np.float64)
        w = (x - self.mean_) @ self.whiten_
        for p, x_sorted, g_sorted in self.steps_:
            proj = w @ p
            gauss = _interp_extrap(proj, x_sorted, g_sorted)
            w = w + np.outer(gauss - proj, p)
        return w

    def inverse(self, y: np.ndarray) -> np.ndarray:
        """Map Gaussian factors ``y`` back to the original data space."""
        y = np.asarray(y, dtype=np.float64)
        w = y.copy()
        for p, x_sorted, g_sorted in reversed(self.steps_):
            proj_g = w @ p
            proj_x = _interp_extrap(proj_g, g_sorted, x_sorted, clip=True)
            w = w + np.outer(proj_x - proj_g, p)
        return self.mean_ + w @ self.dewhiten_


def ppmt_simulate(
    cloud: np.ndarray,
    backbone: np.ndarray,
    n_iter: int = 30,
    n_dirs: int = 60,
    random_state: int | None = None,
) -> np.ndarray:
    """PPMT geostatistical simulation following the target ``cloud``.

    Args:
        cloud: ``(m, d)`` reference target distribution.
        backbone: ``(n, d)`` independent spatially-correlated Gaussian fields
            (same FFT-MA backbone the MST path uses; for conditional PPMT, pass
            a backbone already conditioned on the Gaussianized hard data).
        n_iter, n_dirs: PPMT parameters.
        random_state: PPMT seed.

    Returns:
        ``(n, d)`` simulated field obtained by inverse-PPMT of the backbone.
    """
    cloud = np.asarray(cloud, dtype=np.float64)
    backbone = np.asarray(backbone, dtype=np.float64)
    ppmt = PPMT(n_iter=n_iter, n_dirs=n_dirs, random_state=random_state).fit(cloud)
    # Map each backbone column (by rank) onto the empirical marginal of the
    # corresponding forward factor, so the simulated factors match the
    # distribution PPMT expects to invert. (Standardizing alone leaves a
    # marginal mismatch that inflates the back-transformed variance.)
    g = np.empty_like(backbone)
    for c in range(backbone.shape[1]):
        ranks = (np.argsort(np.argsort(backbone[:, c])) + 0.5) / backbone.shape[0]
        f_sorted = np.sort(ppmt.factors_[:, c])
        idx = np.clip((ranks * len(f_sorted)).astype(int), 0, len(f_sorted) - 1)
        g[:, c] = f_sorted[idx]
    return ppmt.inverse(g)
