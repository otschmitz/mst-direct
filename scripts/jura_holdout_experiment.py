"""Blind conditional hold-out experiment on Swiss Jura trace metals.

The official 259 training / 100 validation split is never modified. Validation
concentrations are loaded only after every realization has been generated.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
import pyreadr
import scipy
from scipy.spatial.distance import cdist
from scipy.stats import norm, rankdata

from mst_direct import PPMT, ScalableMST, empirical_variogram, fit_spherical
from mst_direct import transport_features
from mst_direct.variograms import spherical


ROOT = Path(__file__).resolve().parents[1]
VARIABLES = ("Cd", "Co", "Cr", "Cu", "Ni", "Pb", "Zn")
N_REALIZATIONS = 30


def covariance(coords_a: np.ndarray, coords_b: np.ndarray, rng: float) -> np.ndarray:
    distances = cdist(coords_a, coords_b)
    return 1.0 - spherical(distances, sill=1.0, rng=rng)


def fitted_ranges(values: np.ndarray, coords: np.ndarray) -> list[float]:
    distances = cdist(coords, coords)
    max_dist = float(np.percentile(distances[np.triu_indices(len(coords), 1)], 60))
    ranges = []
    for c in range(values.shape[1]):
        lags, gamma, _ = empirical_variogram(
            values[:, c], coords, n_lags=12, max_dist=max_dist
        )
        fitted, _ = fit_spherical(lags, gamma, sill=float(np.var(values[:, c])))
        ranges.append(float(np.clip(fitted, max_dist / 20, max_dist * 1.5)))
    return ranges


class IrregularConditionalSampler:
    """Reusable simple-kriging conditional Gaussian sampler."""

    def __init__(self, all_coords: np.ndarray, n_train: int, ranges: list[float]):
        self.n_train = n_train
        train_coords = all_coords[:n_train]
        self.operators = []
        self.cholesky = []
        for rng in ranges:
            c_all = covariance(all_coords, all_coords, rng)
            c_train = covariance(train_coords, train_coords, rng)
            c_train.flat[:: n_train + 1] += 1e-7
            self.operators.append(
                covariance(all_coords, train_coords, rng)
                @ np.linalg.inv(c_train)
            )
            c_all.flat[:: len(all_coords) + 1] += 1e-7
            self.cholesky.append(np.linalg.cholesky(c_all))

    def draw(self, hard_values: np.ndarray, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        out = np.empty((self.cholesky[0].shape[0], len(self.cholesky)))
        for c, (chol, weights) in enumerate(zip(self.cholesky, self.operators)):
            unconditional = chol @ rng.standard_normal(len(out))
            out[:, c] = unconditional + weights @ (
                hard_values[:, c] - unconditional[: self.n_train]
            )
        out[: self.n_train] = hard_values
        return out


def normal_scores(x: np.ndarray) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    transformed = np.empty_like(x)
    tables = []
    for c in range(x.shape[1]):
        order = np.argsort(x[:, c], kind="stable")
        probabilities = (np.arange(len(x)) + 0.5) / len(x)
        gaussian = norm.ppf(probabilities)
        transformed[order, c] = gaussian
        tables.append((gaussian, x[order, c]))
    return transformed, tables


def inverse_normal_scores(z: np.ndarray, tables) -> np.ndarray:
    return np.column_stack([
        np.interp(z[:, c], table[0], table[1]) for c, table in enumerate(tables)
    ])


def ensemble_metrics(ensemble: np.ndarray, observed: np.ndarray, train: np.ndarray) -> dict:
    # Metrics in standardized log-concentration space give all seven metals
    # comparable influence. Raw-unit marginal metrics are also retained.
    center, scale = train.mean(0), train.std(0)
    standardized = (ensemble - center) / scale
    observed_std = (observed - center) / scale
    mean_prediction = ensemble.mean(0)
    errors = mean_prediction - observed
    lower, upper = np.quantile(ensemble, [0.05, 0.95], axis=0)
    coverage = ((observed >= lower) & (observed <= upper)).mean(0)
    width = (upper - lower).mean(0)

    crps = []
    for c in range(ensemble.shape[2]):
        first = np.mean(np.abs(ensemble[:, :, c] - observed[None, :, c]), axis=0)
        pairwise = np.mean(
            np.abs(ensemble[:, None, :, c] - ensemble[None, :, :, c]), axis=(0, 1)
        )
        crps.append(float(np.mean(first - 0.5 * pairwise)))

    energy = []
    for i in range(observed.shape[0]):
        draws = standardized[:, i, :]
        first = np.linalg.norm(draws - observed_std[i], axis=1).mean()
        second = cdist(draws, draws).mean()
        energy.append(first - 0.5 * second)

    ranks = np.sum(ensemble < observed[None, :, :], axis=0)
    return {
        "rmse_log": np.sqrt(np.mean(errors**2, axis=0)).tolist(),
        "mae_log": np.mean(np.abs(errors), axis=0).tolist(),
        "coverage_90": coverage.tolist(),
        "coverage_90_mean": float(coverage.mean()),
        "interval_width_log": width.tolist(),
        "crps_log": crps,
        "crps_log_mean": float(np.mean(crps)),
        "energy_score_standardized_mean": float(np.mean(energy)),
        "rank_histogram": [
            np.bincount(ranks[:, c], minlength=ensemble.shape[0] + 1).tolist()
            for c in range(ensemble.shape[2])
        ],
    }


def standardized_site_energy(
    ensemble: np.ndarray, observed: np.ndarray, train: np.ndarray
) -> np.ndarray:
    center, scale = train.mean(0), train.std(0)
    draws = (ensemble - center) / scale
    truth = (observed - center) / scale
    scores = np.empty(len(observed))
    for i in range(len(observed)):
        first = np.linalg.norm(draws[:, i, :] - truth[i], axis=1).mean()
        scores[i] = first - 0.5 * cdist(draws[:, i, :], draws[:, i, :]).mean()
    return scores


def paired_bootstrap_difference(a: np.ndarray, b: np.ndarray, seed: int) -> dict:
    """Mean paired difference a-b and percentile CI over validation sites."""
    rng = np.random.default_rng(seed)
    difference = a - b
    indices = rng.integers(0, len(difference), size=(5000, len(difference)))
    replicates = difference[indices].mean(axis=1)
    return {
        "mean": float(difference.mean()),
        "ci95": np.quantile(replicates, [0.025, 0.975]).tolist(),
        "interpretation": "negative favors the first named method",
    }


def main() -> None:
    loaded = pyreadr.read_r(str(ROOT / "data" / "jura.rda"))
    train_frame = loaded["jura.pred"]
    # Deliberately do not access jura.val concentrations until generation ends.
    validation_locations = loaded["jura.val"][["Xloc", "Yloc", "Rock"]].copy()
    train_coords = train_frame[["Xloc", "Yloc"]].to_numpy(float)
    validation_coords = validation_locations[["Xloc", "Yloc"]].to_numpy(float)
    all_coords = np.vstack((train_coords, validation_coords))
    train_raw = train_frame[list(VARIABLES)].to_numpy(float)
    if np.any(train_raw <= 0):
        raise ValueError("Log transform requires positive training concentrations")
    train = np.log(train_raw)
    n_train, n_validation = len(train), len(validation_coords)

    # PPMT and independent normal-score baselines are fitted once, on training.
    ppmt = PPMT(n_iter=30, n_dirs=80, random_state=731).fit(train)
    ppmt_ranges = fitted_ranges(ppmt.factors_, train_coords)
    ppmt_sampler = IrregularConditionalSampler(all_coords, n_train, ppmt_ranges)
    ns_train, ns_tables = normal_scores(train)
    ns_ranges = fitted_ranges(ns_train, train_coords)
    ns_sampler = IrregularConditionalSampler(all_coords, n_train, ns_ranges)

    ensembles = {"mst_direct": [], "ppmt": [], "independent_nscore": []}
    diagnostics = []
    started_all = time.perf_counter()
    for realization in range(N_REALIZATIONS):
        seed = 10_000 + realization
        rng = np.random.default_rng(seed)

        # Target cloud: all hard tuples plus 100 training-only bootstrap tuples.
        bootstrap = train[rng.integers(0, n_train, size=n_validation)]
        target = np.vstack((train, bootstrap))
        latent = transport_features(target, "whiten")
        latent_ranges = fitted_ranges(latent[:n_train], train_coords)
        mst_sampler = IrregularConditionalSampler(all_coords, n_train, latent_ranges)
        backbone = mst_sampler.draw(latent[:n_train], seed)
        model = ScalableMST(
            beta=6, n_candidates=48, max_candidates=96,
            candidate_zero_tolerance=0.01, n_iter=150,
            feature_transform="whiten", random_state=seed,
        )
        began = time.perf_counter()
        mst = model.simulate(
            backbone, target,
            pinned=(np.arange(n_train), np.arange(n_train)),
            coords=all_coords,
        )
        elapsed = time.perf_counter() - began
        hard_error = float(np.max(np.abs(mst[:n_train] - train)))
        ensembles["mst_direct"].append(mst[n_train:])
        diagnostics.append({
            "realization": realization,
            "seed": seed,
            "match_seconds": elapsed,
            "hard_data_max_abs_error_log": hard_error,
            **model.diagnostics_,
        })

        ppmt_backbone = ppmt_sampler.draw(ppmt.factors_, seed + 100_000)
        ensembles["ppmt"].append(ppmt.inverse(ppmt_backbone)[n_train:])
        ns_backbone = ns_sampler.draw(ns_train, seed + 200_000)
        ensembles["independent_nscore"].append(
            inverse_normal_scores(ns_backbone, ns_tables)[n_train:]
        )

    # Validation values are first materialized here, after all models are frozen.
    validation_frame = pyreadr.read_r(str(ROOT / "data" / "jura.rda"))["jura.val"]
    validation_raw = validation_frame[list(VARIABLES)].to_numpy(float)
    validation = np.log(validation_raw)
    results = {
        "status": "completed blind hold-out experiment",
        "dataset": {
            "name": "Swiss Jura trace-metal geochemistry",
            "training_count": n_train,
            "validation_count": n_validation,
            "variables": list(VARIABLES),
            "transform": "natural logarithm fitted without validation values",
            "source_commit": "c4612f3abcdb58b79f5639c2562d8314ae657ce8",
            "sha256": "f1d9e8a7e6686aa7473a1ce452fffe74f0c45a3e2ba464e35a9163c0d447d845",
        },
        "protocol": {
            "realizations": N_REALIZATIONS,
            "target_cloud": "259 training tuples + 100 training-only bootstrap tuples",
            "validation_used_for_tuning": False,
            "mst_parameters": {"beta": 6, "initial_candidates": 48, "max_candidates": 96},
        },
        "environment": {
            "python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "pyreadr": pyreadr.__version__,
            "platform": platform.platform(),
        },
        "wall_seconds_total": time.perf_counter() - started_all,
        "mst_diagnostics": diagnostics,
        "metrics": {},
        "paired_energy_score_differences": {},
        "energy_score_by_rock": {},
    }
    site_energy = {}
    for method, draws in ensembles.items():
        ensemble = np.asarray(draws)
        results["metrics"][method] = ensemble_metrics(ensemble, validation, train)
        site_energy[method] = standardized_site_energy(ensemble, validation, train)
        groups = {}
        rocks = validation_locations["Rock"].astype(str).to_numpy()
        for rock in sorted(np.unique(rocks)):
            mask = rocks == rock
            groups[rock] = {
                "count": int(mask.sum()),
                "mean": float(site_energy[method][mask].mean()),
            }
        results["energy_score_by_rock"][method] = groups
    results["paired_energy_score_differences"] = {
        "mst_direct_minus_ppmt": paired_bootstrap_difference(
            site_energy["mst_direct"], site_energy["ppmt"], 9191
        ),
        "mst_direct_minus_independent_nscore": paired_bootstrap_difference(
            site_energy["mst_direct"], site_energy["independent_nscore"], 9292
        ),
    }
    target = ROOT / "jura_holdout_results.json"
    target.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
