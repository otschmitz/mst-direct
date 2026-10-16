"""Single-seed 40k validation after the corrected 10k screening audit."""

import json
import time
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from mst_direct import ScalableMST, gaussian_backbone, grid_coords, sampled_variogram


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    raw = loadmat(ROOT / "data" / "reference_data.mat")
    cloud = np.column_stack([raw[f"z{i}_analytic"].ravel() for i in range(1, 7)])
    coords = grid_coords(200)
    backbone = gaussian_backbone((200, 200), 6, 40, random_state=1)
    model = ScalableMST(
        beta=6, n_candidates=48, max_candidates=96,
        candidate_zero_tolerance=0.01, n_iter=150,
        feature_transform="whiten", random_state=1,
    )
    started = time.perf_counter()
    simulated = model.simulate(backbone, cloud, coords=coords)
    elapsed = time.perf_counter() - started
    evaluations = []
    for evaluation_seed in (101, 202, 303):
        corr, nrmse, sill = [], [], []
        for c in range(6):
            _, gb, cb = sampled_variogram(
                backbone[:, c], coords, n_lags=16, max_dist=100,
                n_pairs=160_000, random_state=evaluation_seed,
            )
            _, gs, cs = sampled_variogram(
                simulated[:, c], coords, n_lags=16, max_dist=100,
                n_pairs=160_000, random_state=evaluation_seed,
            )
            ok = np.isfinite(gb) & np.isfinite(gs)
            corr.append(float(np.corrcoef(gb[ok], gs[ok])[0, 1]))
            bs = max(float(np.nanmax(gb[ok])), 1e-12)
            ss = max(float(np.nanmax(gs[ok])), 1e-12)
            nrmse.append(float(np.sqrt(np.mean((gb[ok] / bs - gs[ok] / ss) ** 2))))
            sill.append(float(ss / max(float(np.var(cloud[:, c])), 1e-12)))
        evaluations.append({
            "evaluation_seed": evaluation_seed,
            "correlations": corr,
            "correlation_min": min(corr),
            "normalized_rmse_max": max(nrmse),
            "sill_to_cloud_variance": sill,
            "pairs_per_lag_bin": int(min(np.minimum(cb, cs))),
        })
    report = {
        "status": "corrected 40k single-seed validation",
        "simulation_seed": 1,
        "match_seconds": elapsed,
        "assignment_is_permutation": bool(
            np.array_equal(np.sort(model.assignment_), np.arange(len(cloud)))
        ),
        "diagnostics": model.diagnostics_,
        "evaluations": evaluations,
    }
    (ROOT / "corrected_40k_validation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
