"""Plotting helpers for inspecting multivariate simulations.

These reproduce the DMS-style validation figures: the marginal + bivariate
histogram matrix, per-variable realization maps, a (methods x variables) map
grid, experimental variograms, and a measured-vs-simulated data-honoring
cross-plot.

``matplotlib`` is an *optional* dependency (``pip install mst-direct[plot]``);
this module is therefore **not** imported by :mod:`mst_direct` at package load.
Import it explicitly::

    from mst_direct.plots import plot_scatter_matrix
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .metrics import _shared_ranges
from .variograms import sampled_variogram, spherical

__all__ = [
    "plot_scatter_matrix",
    "plot_realization",
    "plot_realization_grid",
    "plot_variograms",
    "plot_data_honoring",
]


def plot_scatter_matrix(data, bins=60, ranges=None, title=None, labels=None):
    """``d x d`` marginal (diagonal) + bivariate (off-diagonal) histogram matrix.

    Pass a fixed ``ranges`` (e.g. from a reference) so multiple figures are
    directly comparable. Returns the matplotlib ``Figure``.
    """
    data = np.asarray(data, dtype=np.float64)
    d = data.shape[1]
    if ranges is None:
        ranges = _shared_ranges(data, bins)
    labels = labels or [f"$z_{{{i+1}}}$" for i in range(d)]

    fig, axes = plt.subplots(d, d, figsize=(2.0 * d, 2.0 * d))
    for i in range(d):
        for j in range(d):
            ax = axes[i, j]
            if i == j:
                ax.hist(data[:, i], bins=bins, range=ranges[i], color="#3b6ea5")
            else:
                ax.hist2d(data[:, j], data[:, i], bins=bins,
                          range=[ranges[j], ranges[i]], cmap="viridis")
            ax.set_xticks([])
            ax.set_yticks([])
            if i == d - 1:
                ax.set_xlabel(labels[j])
            if j == 0:
                ax.set_ylabel(labels[i])
    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98 if title else 1))
    return fig


def plot_realization(field, shape, title=None, labels=None, data_coords=None):
    """Per-variable 2-D maps of a realization; overlay data dots if given."""
    field = np.asarray(field, dtype=np.float64)
    ny, nx = shape
    d = field.shape[1]
    labels = labels or [f"$z_{{{i+1}}}$" for i in range(d)]
    ncol = 3
    nrow = int(np.ceil(d / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.5 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for c in range(d):
        im = axes[c].imshow(field[:, c].reshape(ny, nx), origin="lower",
                            cmap="turbo", aspect="equal")
        if data_coords is not None:
            axes[c].scatter(data_coords[:, 0], data_coords[:, 1], s=6, c="k",
                            edgecolors="white", linewidths=0.3)
        axes[c].set_title(labels[c])
        axes[c].set_xticks([])
        axes[c].set_yticks([])
        fig.colorbar(im, ax=axes[c], fraction=0.046, pad=0.04)
    for c in range(d, len(axes)):
        axes[c].axis("off")
    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97 if title else 1))
    return fig


def plot_realization_grid(fields, shape, data_coords=None, title=None, var_labels=None):
    """Compare realizations as a (methods x variables) map grid.

    Rows are the methods (e.g. reference / PPMT / MST-Direct), columns the
    variables; the colour scale is shared per column for fair comparison.
    """
    methods = list(fields)
    ny, nx = shape
    d = next(iter(fields.values())).shape[1]
    var_labels = var_labels or [f"$z_{{{i+1}}}$" for i in range(d)]
    nrow, ncol = len(methods), d
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.3 * ncol, 2.5 * nrow))
    axes = np.atleast_2d(axes)
    for c in range(d):
        allv = np.concatenate([fields[m][:, c] for m in methods])
        vmin, vmax = np.percentile(allv, [1, 99])
        im = None
        for r, m in enumerate(methods):
            ax = axes[r, c]
            im = ax.imshow(fields[m][:, c].reshape(ny, nx), origin="lower",
                           cmap="turbo", vmin=vmin, vmax=vmax, aspect="equal")
            if data_coords is not None:
                ax.scatter(data_coords[:, 0], data_coords[:, 1], s=3, c="k",
                           edgecolors="white", linewidths=0.2)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(var_labels[c])
            if c == 0:
                ax.set_ylabel(m, fontsize=11)
        fig.colorbar(im, ax=axes[:, c], location="bottom", fraction=0.05, pad=0.03)
    if title:
        fig.suptitle(title, fontsize=14)
    return fig


def plot_data_honoring(measured, simulated_at_data, title=None, labels=None):
    """Cross-plot of measured vs simulated hard-data values (45-degree line)."""
    measured = np.asarray(measured, dtype=np.float64)
    simulated_at_data = np.asarray(simulated_at_data, dtype=np.float64)
    d = measured.shape[1]
    labels = labels or [f"$z_{{{i+1}}}$" for i in range(d)]
    ncol = 3
    nrow = int(np.ceil(d / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.4 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for c in range(d):
        ax = axes[c]
        lo = min(measured[:, c].min(), simulated_at_data[:, c].min())
        hi = max(measured[:, c].max(), simulated_at_data[:, c].max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.scatter(measured[:, c], simulated_at_data[:, c], s=12, c="tab:blue", alpha=0.7)
        err = float(np.max(np.abs(measured[:, c] - simulated_at_data[:, c])))
        ax.set_title(f"{labels[c]}  (max err = {err:.1e})")
        ax.set_xlabel("measured")
        ax.set_ylabel("simulated")
    for c in range(d, len(axes)):
        axes[c].axis("off")
    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96 if title else 1))
    return fig


def plot_variograms(fields, coords, max_dist, colors=None, title=None,
                    n_lags=18, n_pairs=300_000, target_range=None, target_sills=None):
    """Experimental variograms of several fields per variable.

    If ``target_range`` and ``target_sills`` are given, overlay the imposed
    spherical model (dashed) as the target.
    """
    labels = list(fields)
    d = next(iter(fields.values())).shape[1]
    colors = colors or {}
    ncol = 3
    nrow = int(np.ceil(d / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for c in range(d):
        ax = axes[c]
        for lab in labels:
            cen, g, _ = sampled_variogram(fields[lab][:, c], coords,
                                          n_lags=n_lags, max_dist=max_dist, n_pairs=n_pairs)
            ax.plot(cen, g, marker="o", ms=3, label=lab, color=colors.get(lab))
        if target_range is not None and target_sills is not None:
            hh = np.linspace(0, max_dist, 100)
            ax.plot(hh, spherical(hh, sill=float(target_sills[c]), rng=target_range),
                    "k--", lw=1.2, label=f"target (sph r={target_range:.0f})")
        ax.set_title(f"$z_{{{c+1}}}$")
        ax.set_xlabel("lag")
        ax.set_ylabel(r"$\gamma$")
        if c == 0:
            ax.legend(fontsize=8)
    for c in range(d, len(axes)):
        axes[c].axis("off")
    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96 if title else 1))
    return fig
