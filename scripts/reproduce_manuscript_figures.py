"""Reproduce the DMS manuscript figures with the corrected implementation.

This script replaces the unavailable historical plotting workflow.  It uses
the checksum-verified DMS inputs, fixed seeds, the corrected sparse matcher,
and the repository's documented PPMT comparator.  It writes publication-grade
PNG/PDF figures, machine-readable metrics, and compressed numerical results.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.io import loadmat

from mst_direct import (
    PPMT,
    ScalableMST,
    conditional_gaussian_backbone,
    gaussian_backbone,
    grid_coords,
    histogram_mse_table,
    ppmt_simulate,
    sampled_variogram,
    spherical,
    transport_features,
)
from mst_direct.metrics import _shared_ranges
from mst_direct.plots import (
    plot_data_honoring,
    plot_realization_grid,
    plot_scatter_matrix,
)


ROOT = Path(__file__).resolve().parents[1]


def _save(fig: plt.Figure, stem: Path) -> None:
    """Save an editable PDF and a high-resolution submission PNG."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=360, bbox_inches="tight")
    plt.close(fig)


def _plot_variograms(
    fields: dict[str, np.ndarray],
    coords: np.ndarray,
    max_dist: float,
    stem: Path,
    target_range: float | None = None,
) -> dict[str, list[list[float]]]:
    """Plot lag-stratified variograms using identical pairs for every field."""
    colors = {"Reference": "#222222", "MST-Direct": "#0072B2", "PPMT": "#D55E00"}
    d = next(iter(fields.values())).shape[1]
    fig, axes = plt.subplots(2, 3, figsize=(11.4, 6.6), constrained_layout=True)
    axes = axes.ravel()
    values: dict[str, list[list[float]]] = {name: [] for name in fields}
    for c in range(d):
        ax = axes[c]
        for name, field in fields.items():
            centres, gamma, _ = sampled_variogram(
                field[:, c], coords, n_lags=16, max_dist=max_dist,
                n_pairs=160_000, random_state=2026,
                sampling="lag_stratified",
            )
            values[name].append(gamma.tolist())
            ax.plot(centres, gamma, marker="o", markersize=3.2, linewidth=1.35,
                    color=colors[name], label=name)
        if target_range is not None:
            h = np.linspace(0, max_dist, 200)
            sill = float(np.var(fields["MST-Direct"][:, c]))
            ax.plot(h, spherical(h, sill=sill, rng=target_range), "k--",
                    linewidth=1.0, label=f"Target (range {target_range:g})")
        ax.set(title=rf"$z_{{{c + 1}}}$", xlabel="Lag distance", ylabel=r"Semivariance $\gamma(h)$")
        ax.grid(alpha=0.2, linewidth=0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=len(labels), frameon=False)
    _save(fig, stem)
    return values


def _load_unconditional() -> np.ndarray:
    raw = loadmat(ROOT / "data" / "reference_data.mat")
    return np.column_stack([raw[f"z{i}_analytic"].ravel() for i in range(1, 7)])


def _load_conditional() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = loadmat(ROOT / "data" / "HardData_ReferenceModel_size100_range20.mat")
    # MATLAB stores the model as (variable, x, y); convert to row-major (y, x, variable).
    reference = np.moveaxis(raw["reference_models"], 0, -1).transpose(1, 0, 2)
    positions = raw["cond_pos"].astype(np.int64) - 1
    data_idx = positions[:, 1] * 100 + positions[:, 0]
    cloud = reference.reshape(-1, 6)
    measured = raw["cond_value"].astype(np.float64)
    if not np.array_equal(cloud[data_idx], measured):
        raise RuntimeError("DMS conditioning coordinates do not match the reference model")
    return cloud, data_idx, measured


def _run_unconditional(figures: Path) -> tuple[dict, dict[str, np.ndarray]]:
    cloud = _load_unconditional()
    coords = grid_coords(200)
    backbone = gaussian_backbone((200, 200), 6, 40, random_state=1)

    model = ScalableMST(
        beta=6, n_candidates=48, max_candidates=96,
        candidate_zero_tolerance=0.01, n_iter=150,
        feature_transform="whiten", random_state=1,
    )
    started = time.perf_counter()
    mst = model.simulate(backbone, cloud, coords=coords)
    mst_seconds = time.perf_counter() - started

    started = time.perf_counter()
    ppmt = ppmt_simulate(cloud, backbone, n_iter=30, n_dirs=60, random_state=1)
    ppmt_seconds = time.perf_counter() - started

    ranges = _shared_ranges(cloud, 60)
    for name, values in (("uncond_reference_scatter", cloud),
                         ("uncond_ppmt_scatter", ppmt),
                         ("uncond_mst_scatter", mst)):
        _save(plot_scatter_matrix(values, bins=60, ranges=ranges), figures / name)
    variograms = _plot_variograms(
        {"MST-Direct": mst, "PPMT": ppmt}, coords, 100,
        figures / "uncond_variograms", target_range=40,
    )
    result = {
        "grid": [200, 200], "seed": 1, "range": 40,
        "mst_seconds": mst_seconds, "ppmt_seconds": ppmt_seconds,
        "mst_diagnostics": model.diagnostics_,
        "mst_is_permutation": bool(np.array_equal(np.sort(model.assignment_), np.arange(len(cloud)))),
        "histogram_mse_mst": histogram_mse_table(cloud, mst, bins=50).tolist(),
        "histogram_mse_ppmt": histogram_mse_table(cloud, ppmt, bins=50).tolist(),
        "variograms": variograms,
    }
    return result, {"reference": cloud, "backbone": backbone, "mst": mst, "ppmt": ppmt}


def _run_conditional(figures: Path) -> tuple[dict, dict[str, np.ndarray]]:
    cloud, data_idx, measured = _load_conditional()
    coords = grid_coords(100)
    range_value = 16.0

    latent = transport_features(cloud, "whiten")
    backbone_mst = conditional_gaussian_backbone(
        (100, 100), coords, data_idx, latent[data_idx], range_value, random_state=2,
    )
    model = ScalableMST(
        beta=6, n_candidates=48, max_candidates=96,
        candidate_zero_tolerance=0.01, n_iter=150,
        feature_transform="whiten", random_state=2,
    )
    started = time.perf_counter()
    mst = model.simulate(
        backbone_mst, cloud, pinned=(data_idx, data_idx), coords=coords,
    )
    mst_seconds = time.perf_counter() - started

    transform = PPMT(n_iter=30, n_dirs=60, random_state=2).fit(cloud)
    factors = transform.forward(cloud)
    backbone_ppmt = conditional_gaussian_backbone(
        (100, 100), coords, data_idx, factors[data_idx], range_value, random_state=2,
    )
    started = time.perf_counter()
    ppmt = transform.inverse(backbone_ppmt)
    ppmt_seconds = time.perf_counter() - started

    ranges = _shared_ranges(cloud, 60)
    for name, values in (("cond_reference_scatter", cloud),
                         ("cond_ppmt_scatter", ppmt),
                         ("cond_mst_scatter", mst)):
        _save(plot_scatter_matrix(values, bins=60, ranges=ranges), figures / name)
    positions = coords[data_idx]
    _save(plot_realization_grid(
        {"Reference": cloud, "PPMT": ppmt, "MST-Direct": mst}, (100, 100),
        data_coords=positions,
    ), figures / "cond_maps_comparison")
    _save(plot_data_honoring(measured, mst[data_idx]), figures / "cond_data_honoring")
    variograms = _plot_variograms(
        {"Reference": cloud, "MST-Direct": mst, "PPMT": ppmt}, coords, 50,
        figures / "cond_variograms",
    )
    result = {
        "grid": [100, 100], "seed": 2, "range": range_value,
        "hard_data_count": int(len(data_idx)),
        "mst_seconds": mst_seconds, "ppmt_inverse_seconds": ppmt_seconds,
        "mst_diagnostics": model.diagnostics_,
        "mst_is_permutation": bool(np.array_equal(np.sort(model.assignment_), np.arange(len(cloud)))),
        "mst_hard_data_max_abs_error": float(np.max(np.abs(mst[data_idx] - measured))),
        "ppmt_hard_data_max_abs_error": float(np.max(np.abs(ppmt[data_idx] - measured))),
        "histogram_mse_mst": histogram_mse_table(cloud, mst, bins=50).tolist(),
        "histogram_mse_ppmt": histogram_mse_table(cloud, ppmt, bins=50).tolist(),
        "variograms": variograms,
    }
    return result, {
        "reference": cloud, "data_idx": data_idx, "measured": measured,
        "backbone_mst": backbone_mst, "backbone_ppmt": backbone_ppmt,
        "mst": mst, "ppmt": ppmt,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "manuscript_reproduction")
    args = parser.parse_args()
    output = args.output.resolve()
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    unconditional, uncond_arrays = _run_unconditional(figures)
    conditional, cond_arrays = _run_conditional(figures)
    report = {
        "status": "independent deterministic reproduction with corrected implementation",
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "numpy": np.__version__, "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "configuration": {
            "mst": {"beta": 6, "initial_candidates": 48, "max_candidates": 96,
                    "sinkhorn_iterations": 150, "feature_transform": "whiten"},
            "ppmt": {"iterations": 30, "directions_per_iteration": 60},
            "figures": {"png_dpi": 360, "editable_pdf": True},
        },
        "unconditional": unconditional,
        "conditional": conditional,
    }
    (output / "manuscript_reproduction.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8",
    )
    np.savez_compressed(output / "unconditional_results.npz", **uncond_arrays)
    np.savez_compressed(output / "conditional_results.npz", **cond_arrays)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
