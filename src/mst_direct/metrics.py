"""Evaluation metrics: shape preservation and variogram preservation.

Reproduces the two metrics reported in the paper:

* **Shape preservation** (eq. 14) — 2-D histogram similarity of the joint
  distribution, ``S_H2D = sum_ij min(H_orig(i,j), H_sim(i,j))`` for normalized
  histograms. Ranges in ``[0, 1]``; ``1`` means the joint distribution is
  reproduced exactly.
* **Variogram preservation** (eq. 15) — Pearson correlation of empirical
  variograms (re-exported from :mod:`mst_direct.variograms`).
"""

from __future__ import annotations

import numpy as np

from .variograms import variogram_correlation

__all__ = [
    "histogram2d_similarity",
    "shape_preservation",
    "variogram_correlation",
    "histogram_mse_table",
]


def _shared_ranges(reference: np.ndarray, bins: int):
    return [
        (float(reference[:, c].min()), float(reference[:, c].max()))
        for c in range(reference.shape[1])
    ]


def histogram2d_similarity(
    original: np.ndarray,
    simulated: np.ndarray,
    bins: int = 20,
    value_range: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> float:
    """2-D histogram similarity of two bivariate samples (eq. 14).

    Both samples are binned on a *shared* grid (so the comparison is
    meaningful), normalized to sum to 1, and compared via histogram
    intersection ``sum min(H_orig, H_sim)``.

    Args:
        original: ``(n, 2)`` original joint sample.
        simulated: ``(n, 2)`` simulated joint sample.
        bins: Number of bins per axis.
        value_range: Optional ``((xmin, xmax), (ymin, ymax))``. Defaults to the
            combined range of both samples.

    Returns:
        Histogram-intersection similarity in ``[0, 1]``.
    """
    original = np.asarray(original, dtype=np.float64)
    simulated = np.asarray(simulated, dtype=np.float64)
    if original.shape[1] != 2 or simulated.shape[1] != 2:
        raise ValueError("histogram2d_similarity expects bivariate (n, 2) inputs")

    if value_range is None:
        both = np.vstack([original, simulated])
        value_range = (
            (float(both[:, 0].min()), float(both[:, 0].max())),
            (float(both[:, 1].min()), float(both[:, 1].max())),
        )

    h_orig, _, _ = np.histogram2d(
        original[:, 0], original[:, 1], bins=bins, range=value_range
    )
    h_sim, _, _ = np.histogram2d(
        simulated[:, 0], simulated[:, 1], bins=bins, range=value_range
    )

    s_orig = h_orig.sum()
    s_sim = h_sim.sum()
    if s_orig == 0 or s_sim == 0:
        return 0.0
    h_orig = h_orig / s_orig
    h_sim = h_sim / s_sim
    return float(np.minimum(h_orig, h_sim).sum())


def shape_preservation(
    original: np.ndarray,
    simulated: np.ndarray,
    bins: int = 20,
    value_range: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> float:
    """Alias for :func:`histogram2d_similarity` (the paper's S_H2D metric)."""
    return histogram2d_similarity(original, simulated, bins=bins, value_range=value_range)


def histogram_mse_table(
    reference: np.ndarray,
    simulated: np.ndarray,
    bins: int = 50,
) -> np.ndarray:
    """``(d, d)`` MSE between simulated and reference experimental histograms.

    Diagonal entries compare 1-D marginal histograms; off-diagonal entries
    compare 2-D bivariate histograms. Histograms share the reference's value
    range and are normalized to sum to one. This is the multivariate
    reproduction metric of the v2 experiments (DMS-style Tables 1-4).
    """
    reference = np.asarray(reference, dtype=np.float64)
    simulated = np.asarray(simulated, dtype=np.float64)
    d = reference.shape[1]
    ranges = _shared_ranges(reference, bins)
    table = np.zeros((d, d))
    for i in range(d):
        hr, edges = np.histogram(reference[:, i], bins=bins, range=ranges[i], density=False)
        hs, _ = np.histogram(simulated[:, i], bins=edges)
        hr = hr / hr.sum()
        hs = hs / max(hs.sum(), 1)
        table[i, i] = float(np.mean((hr - hs) ** 2))
        for j in range(i + 1, d):
            Hr, _, _ = np.histogram2d(
                reference[:, i], reference[:, j], bins=bins, range=[ranges[i], ranges[j]]
            )
            Hs, _, _ = np.histogram2d(
                simulated[:, i], simulated[:, j], bins=bins, range=[ranges[i], ranges[j]]
            )
            Hr = Hr / Hr.sum()
            Hs = Hs / max(Hs.sum(), 1)
            mse = float(np.mean((Hr - Hs) ** 2))
            table[i, j] = table[j, i] = mse
    return table
