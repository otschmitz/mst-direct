"""Independent audit rerun on DMS data; not the missing manuscript script."""

import json
import platform
import resource
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.io import loadmat

from mst_direct import (
    ScalableMST,
    conditional_gaussian_backbone,
    gaussian_backbone,
    grid_coords,
    histogram_mse_table,
    transport_features,
)


ROOT = Path(__file__).resolve().parents[1]


def timed_match(backbone, cloud, coords, pinned=None):
    model = ScalableMST(
        beta=6.0,
        n_candidates=48,
        n_iter=150,
        tol=1e-6,
        k_relational=8,
        lam_relational=0.0,
        n_relax=0,
        random_state=42,
    )
    started = time.perf_counter()
    sim = model.simulate(backbone, cloud, pinned=pinned, coords=coords)
    elapsed = time.perf_counter() - started
    assignment = model.assignment_
    return sim, {
        "match_wall_seconds": elapsed,
        "assignment_is_permutation": bool(
            np.array_equal(np.sort(assignment), np.arange(len(assignment)))
        ),
        "histogram_mse_max": float(histogram_mse_table(cloud, sim).max()),
    }


def main():
    ref = loadmat(ROOT / "data" / "reference_data.mat")
    cloud_u = np.column_stack([ref[f"z{i}_analytic"].ravel() for i in range(1, 7)])
    coords_u = grid_coords(200, 200)
    start = time.perf_counter()
    bb_u = gaussian_backbone((200, 200), 6, rng_range=40, random_state=42)
    bb_u_seconds = time.perf_counter() - start
    _, uncond = timed_match(bb_u, cloud_u, coords_u)
    uncond["backbone_wall_seconds"] = bb_u_seconds

    raw = loadmat(ROOT / "data" / "HardData_ReferenceModel_size100_range20.mat")
    cloud_c = np.moveaxis(raw["reference_models"], 0, -1).reshape(-1, 6)
    pos = raw["cond_pos"].astype(np.int64) - 1
    data_rows = pos[:, 0] * 100 + pos[:, 1]
    data_cols = data_rows.copy()
    latent = transport_features(cloud_c, "whiten")
    coords_c = grid_coords(100, 100)
    start = time.perf_counter()
    bb_c = conditional_gaussian_backbone(
        (100, 100), coords_c, data_rows, latent[data_cols],
        rng_range=18, random_state=42,
    )
    bb_c_seconds = time.perf_counter() - start
    sim_c, cond = timed_match(
        bb_c, cloud_c, coords_c, pinned=(data_rows, data_cols)
    )
    cond["backbone_wall_seconds"] = bb_c_seconds
    cond["hard_data_count"] = int(len(data_rows))
    cond["hard_data_max_abs_error"] = float(
        np.max(np.abs(sim_c[data_rows] - raw["cond_value"]))
    )

    report = {
        "status": "independent audit rerun; not original manuscript reproduction",
        "parameters": {
            "beta": 6.0, "n_candidates": 48, "n_iter": 150,
            "relational_weight": 0.0, "relational_passes": 0, "seed": 42,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "peak_rss_kib_process": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
        "unconditional_200x200": uncond,
        "conditional_100x100": cond,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
