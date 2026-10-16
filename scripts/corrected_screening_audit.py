"""10k-node screening audit for the corrected scalable matcher.

This deliberately precedes any expensive 40k rerun.  It separates simulation
seeds from evaluation seeds, uses lag-stratified variograms, and records
candidate/repair diagnostics.
"""

import json
import time
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from mst_direct import ScalableMST, gaussian_backbone, grid_coords, sampled_variogram


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    raw = loadmat(ROOT / "data" / "reference_data.mat")
    full = np.column_stack([raw[f"z{i}_analytic"].ravel() for i in range(1, 7)])
    cloud = full[np.linspace(0, len(full) - 1, 10_000, dtype=np.int64)]
    coords = grid_coords(100)
    records = []
    for simulation_seed in (0, 1, 2):
        backbone = gaussian_backbone((100, 100), 6, 20, random_state=simulation_seed)
        model = ScalableMST(
            beta=6, n_candidates=48, max_candidates=96,
            candidate_zero_tolerance=0.01, n_iter=150,
            feature_transform="whiten", random_state=simulation_seed,
        )
        started = time.perf_counter()
        simulated = model.simulate(backbone, cloud, coords=coords)
        elapsed = time.perf_counter() - started
        by_eval_seed = []
        for evaluation_seed in (101, 202, 303):
            correlations = []
            nrmses = []
            sill_to_cloud_variance = []
            minimum_count = None
            for c in range(6):
                _, gb, cb = sampled_variogram(
                    backbone[:, c], coords, n_lags=16, max_dist=50,
                    n_pairs=160_000, random_state=evaluation_seed,
                    sampling="lag_stratified",
                )
                _, gs, cs = sampled_variogram(
                    simulated[:, c], coords, n_lags=16, max_dist=50,
                    n_pairs=160_000, random_state=evaluation_seed,
                    sampling="lag_stratified",
                )
                ok = np.isfinite(gb) & np.isfinite(gs)
                correlations.append(float(np.corrcoef(gb[ok], gs[ok])[0, 1]))
                backbone_scale = max(float(np.nanmax(gb[ok])), 1e-12)
                simulated_scale = max(float(np.nanmax(gs[ok])), 1e-12)
                nrmses.append(float(np.sqrt(np.mean(
                    (gb[ok] / backbone_scale - gs[ok] / simulated_scale) ** 2
                ))))
                sill_to_cloud_variance.append(float(
                    simulated_scale / max(float(np.var(cloud[:, c])), 1e-12)
                ))
                count = int(np.minimum(cb, cs).min())
                minimum_count = count if minimum_count is None else min(minimum_count, count)
            by_eval_seed.append({
                "evaluation_seed": evaluation_seed,
                "variogram_correlations": correlations,
                "variogram_correlation_min": min(correlations),
                "variogram_nrmse_max": max(nrmses),
                "variogram_sill_to_cloud_variance": sill_to_cloud_variance,
                "minimum_lag_bin_count": minimum_count,
            })
        records.append({
            "simulation_seed": simulation_seed,
            "match_seconds": elapsed,
            "assignment_is_permutation": bool(
                np.array_equal(np.sort(model.assignment_), np.arange(len(cloud)))
            ),
            "diagnostics": model.diagnostics_,
            "evaluations": by_eval_seed,
        })
        print(json.dumps(records[-1]), flush=True)
    (ROOT / "corrected_screening_audit.json").write_text(
        json.dumps({"status": "corrected 10k screening audit", "runs": records}, indent=2)
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
