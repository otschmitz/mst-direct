"""MST-Direct: Matching via Sinkhorn Transport for multivariate geostatistical
simulation.

A multivariate geostatistical simulation method built on entropy-regularized
optimal transport. The target value tuples are *matched* — through the Sinkhorn
algorithm with a relational (k-nearest-neighbor) term — onto a spatial template,
so the simulated field is a spatial *rearrangement* of the target tuples: its
joint distribution is preserved exactly (100% shape preservation) while the
transport reproduces the variogram. This handles complex non-linear
dependencies (bimodal, step, sinusoidal, heteroscedastic) that Gaussian Copula
and LU Decomposition destroy.

**Version 2.0** adds the multivariate / conditional / large-grid capabilities:

* :class:`ScalableMST` — sparse, candidate-restricted Sinkhorn matcher that
  scales to tens of thousands of nodes (``O(n * n_candidates)`` memory), with
  hard-data **conditioning** by pinning;
* :func:`gaussian_backbone` / :func:`conditional_gaussian_backbone` — FFT-MA
  Gaussian templates carrying a prescribed (optionally kriging-conditioned)
  variogram;
* :class:`PPMT` — Projection Pursuit Multivariate Transform comparator;
* scalable variogram estimators (:func:`sampled_variogram`,
  :func:`sampled_cross_variogram`, :func:`fit_spherical`) and the
  :func:`histogram_mse_table` reproduction metric.

References:
    Schmitz, T. B. "MST-Direct: Matching via Sinkhorn Transport for Multivariate
    Geostatistical Simulation with Complex Non-Linear Dependencies."
    arXiv:2603.18036, 2026.

Quick start (v1 — dense, bivariate):
    >>> from mst_direct import MSTDirect, generate_dataset, shape_preservation
    >>> data = generate_dataset("gaussian_mix", grid=(15, 15), random_state=42)
    >>> sim = MSTDirect(random_state=42).simulate(data["values"], data["coords"])
    >>> round(shape_preservation(data["values"], sim), 3)
    1.0

Quick start (v2 — scalable, multivariate, backbone-driven):
    >>> import numpy as np
    >>> from mst_direct import ScalableMST, gaussian_backbone, grid_coords
    >>> N = 24; coords = grid_coords(N, N)
    >>> rng = np.random.default_rng(0)
    >>> cloud = rng.normal(size=(N * N, 3))
    >>> bb = gaussian_backbone((N, N), d=3, rng_range=8.0, random_state=0)
    >>> sim = ScalableMST(random_state=0).simulate(bb, cloud, coords=coords)
    >>> sim.shape
    (576, 3)
"""

from __future__ import annotations

# Submodules (note: ``plots`` is intentionally *not* imported here so the core
# package does not require matplotlib; import ``mst_direct.plots`` explicitly).
from . import (
    adjacency,
    backbone,
    baselines,
    metrics,
    ppmt,
    relational,
    rounding,
    scalable,
    simulator,
    synthetic,
    variograms,
)
from .adjacency import knn_adjacency, pairwise_distances
from .backbone import conditional_gaussian_backbone, gaussian_backbone
from .metrics import histogram2d_similarity, histogram_mse_table, shape_preservation
from .ppmt import PPMT, ppmt_simulate
from .relational import relational_match
from .rounding import greedy_round
from .scalable import ScalableMST, mst_simulate, transport_features
from .simulator import MSTDirect
from .sinkhorn import sinkhorn, sinkhorn_plan
from .synthetic import (
    RELATIONSHIPS,
    apply_relationship,
    fft_ma,
    generate_dataset,
    make_grid,
)
from .variograms import (
    empirical_variogram,
    exponential,
    fit_spherical,
    gaussian,
    grid_coords,
    sampled_cross_variogram,
    sampled_variogram,
    spherical,
    variogram_correlation,
)

__version__ = "2.1.0"

__all__ = [
    "__version__",
    # simulators
    "MSTDirect",          # v1: dense, exact relational matcher
    "ScalableMST",        # v2: sparse candidate-restricted matcher (+conditioning)
    "transport_features",
    "mst_simulate",
    # backbone (v2)
    "gaussian_backbone",
    "conditional_gaussian_backbone",
    # comparator (v2)
    "PPMT",
    "ppmt_simulate",
    # optimal transport
    "sinkhorn",
    "sinkhorn_plan",
    "relational_match",
    "greedy_round",
    # spatial / variograms
    "knn_adjacency",
    "pairwise_distances",
    "spherical",
    "exponential",
    "gaussian",
    "empirical_variogram",
    "variogram_correlation",
    "grid_coords",
    "sampled_variogram",
    "sampled_cross_variogram",
    "fit_spherical",
    # metrics
    "histogram2d_similarity",
    "shape_preservation",
    "histogram_mse_table",
    # synthetic data
    "generate_dataset",
    "make_grid",
    "fft_ma",
    "apply_relationship",
    "RELATIONSHIPS",
    # submodules
    "adjacency",
    "backbone",
    "baselines",
    "metrics",
    "ppmt",
    "relational",
    "rounding",
    "scalable",
    "simulator",
    "synthetic",
    "variograms",
]
