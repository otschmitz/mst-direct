"""Plot the committed Swiss Jura hold-out summary without rerunning models."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LABELS = {"mst_direct": "MST-Direct", "ppmt": "PPMT", "independent_nscore": "Independent NS"}
COLORS = {"mst_direct": "#276FBF", "ppmt": "#D1495B", "independent_nscore": "#6A994E"}


def main() -> None:
    report = json.loads((ROOT / "jura_holdout_results.json").read_text())
    methods = list(LABELS)
    variables = report["dataset"]["variables"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)

    energy = [report["metrics"][m]["energy_score_standardized_mean"] for m in methods]
    axes[0].bar([LABELS[m] for m in methods], energy, color=[COLORS[m] for m in methods])
    axes[0].set_ylabel("Mean energy score")
    axes[0].set_title("Multivariate hold-out score")
    axes[0].tick_params(axis="x", rotation=20)

    x = np.arange(len(variables))
    for method in methods:
        axes[1].plot(x, report["metrics"][method]["coverage_90"], marker="o",
                     label=LABELS[method], color=COLORS[method])
    axes[1].axhline(0.9, color="black", linestyle="--", linewidth=1, label="Nominal")
    axes[1].set_xticks(x, variables)
    axes[1].set_ylim(0.6, 1.0)
    axes[1].set_ylabel("90% interval coverage")
    axes[1].set_title("Marginal calibration")
    axes[1].legend(fontsize=8, ncol=2)

    comparisons = report["paired_energy_score_differences"]
    keys = ["mst_direct_minus_ppmt", "mst_direct_minus_independent_nscore"]
    labels = ["MST - PPMT", "MST - independent NS"]
    means = np.array([comparisons[k]["mean"] for k in keys])
    cis = np.array([comparisons[k]["ci95"] for k in keys])
    axes[2].errorbar(means, np.arange(2),
                     xerr=np.vstack((means - cis[:, 0], cis[:, 1] - means)),
                     fmt="o", color="#333333", capsize=4)
    axes[2].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[2].set_yticks(np.arange(2), labels)
    axes[2].set_xlabel("Paired energy-score difference")
    axes[2].set_title("95% site-bootstrap intervals")
    axes[2].invert_yaxis()

    fig.savefig(ROOT / "jura_holdout_summary.png", dpi=300)


if __name__ == "__main__":
    main()
