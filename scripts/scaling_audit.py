"""Compact independent scaling/sensitivity audit for the released matcher."""

import json
import platform
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.io import loadmat
from scipy.spatial import cKDTree

from mst_direct import ScalableMST, gaussian_backbone, grid_coords, sampled_variogram


ROOT = Path(__file__).resolve().parents[1]


def reference_cloud() -> np.ndarray:
    raw = loadmat(ROOT / "data" / "reference_data.mat")
    return np.column_stack([raw[f"z{i}_analytic"].ravel() for i in range(1, 7)])


def subset(cloud: np.ndarray, n: int, d: int) -> np.ndarray:
    if n == len(cloud):
        return cloud[:, :d].copy()
    idx = np.linspace(0, len(cloud) - 1, n, dtype=np.int64)
    return cloud[idx, :d].copy()


def spatial_metrics(backbone, simulated, coords, seed):
    correlations = []
    normalized_rmse = []
    for c in range(backbone.shape[1]):
        _, gb, _ = sampled_variogram(
            backbone[:, c], coords, n_lags=16, n_pairs=100_000,
            max_dist=np.sqrt(len(coords)) / 2, random_state=seed,
        )
        _, gs, _ = sampled_variogram(
            simulated[:, c], coords, n_lags=16, n_pairs=100_000,
            max_dist=np.sqrt(len(coords)) / 2, random_state=seed,
        )
        mask = np.isfinite(gb) & np.isfinite(gs)
        correlations.append(float(np.corrcoef(gb[mask], gs[mask])[0, 1]))
        gbn = gb[mask] / max(float(np.nanmax(gb[mask])), 1e-12)
        gsn = gs[mask] / max(float(np.nanmax(gs[mask])), 1e-12)
        normalized_rmse.append(float(np.sqrt(np.mean((gbn - gsn) ** 2))))
    return {
        "variogram_shape_correlation_mean": float(np.mean(correlations)),
        "variogram_shape_correlation_min": float(np.min(correlations)),
        "variogram_normalized_rmse_mean": float(np.mean(normalized_rmse)),
        "variogram_normalized_rmse_max": float(np.max(normalized_rmse)),
    }


def run(side: int, d: int, candidates: int, seed: int, cloud_all: np.ndarray):
    n = side * side
    cloud = subset(cloud_all, n, d)
    coords = grid_coords(side, side)
    backbone = gaussian_backbone(
        (side, side), d=d, rng_range=side / 5, random_state=seed
    )
    model = ScalableMST(
        beta=6, n_candidates=candidates, n_iter=150,
        lam_relational=0, n_relax=0, random_state=seed,
    )
    started = time.perf_counter()
    simulated = model.simulate(backbone, cloud, coords=coords)
    elapsed = time.perf_counter() - started

    standardized_cloud = (cloud - cloud.mean(0)) / cloud.std(0)
    standardized_backbone = (backbone - backbone.mean(0)) / backbone.std(0)
    nearest = cKDTree(standardized_cloud).query(
        standardized_backbone, k=min(candidates, n)
    )[1]
    nearest = np.asarray(nearest).reshape(n, min(candidates, n))
    in_original_candidates = np.any(nearest == model.assignment_[:, None], axis=1)

    result = {
        "side": side,
        "nodes": n,
        "dimensions": d,
        "candidates": candidates,
        "seed": seed,
        "match_wall_seconds": elapsed,
        "assignment_is_permutation": bool(
            np.array_equal(np.sort(model.assignment_), np.arange(n))
        ),
        "assigned_outside_original_candidates_fraction": float(
            1.0 - in_original_candidates.mean()
        ),
    }
    result.update(spatial_metrics(backbone, simulated, coords, seed))
    return result


def main():
    cloud = reference_cloud()
    design = []
    # Scaling and seed variability at six dimensions.
    for side in (50, 100, 200):
        for seed in (0, 1, 2):
            design.append((side, 6, 48, seed, "scaling"))
    # Dimensionality at a fixed 10,000-node grid; d=6 already appears above.
    for d in (2, 4):
        design.append((100, d, 48, 0, "dimensions"))
    # Candidate sensitivity; C=48 already appears above.
    for candidates in (24, 96):
        design.append((100, 6, candidates, 0, "candidates"))

    runs = []
    for side, d, candidates, seed, group in design:
        record = run(side, d, candidates, seed, cloud)
        record["design_group"] = group
        runs.append(record)
        print(json.dumps(record), flush=True)

    report = {
        "status": "independent audit; not original manuscript reproduction",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "fixed_parameters": {
            "beta": 6.0, "n_iter": 150,
            "relational_weight": 0.0, "relational_passes": 0,
            "variogram_sampled_pairs_per_variable": 100000,
        },
        "runs": runs,
    }
    print("FINAL_REPORT")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
