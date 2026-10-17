# MST-Direct — Matching via Sinkhorn Transport

**Multivariate geostatistical simulation that preserves the supplied empirical
tuple cloud exactly while approximating a prescribed spatial structure.**

MST-Direct treats the target value tuples as the distribution to reproduce and
finds, via entropy-regularized **optimal transport** (the Sinkhorn algorithm)
with an optional relational *k*-nearest-neighbor refinement, a spatial
arrangement that approximates the backbone variogram. The final realization is
a permutation of the target tuples, so their empirical multivariate
distribution is exactly intact. Spatial reproduction is approximate and depends
on the candidate graph, rounding/completion, seed, and hyperparameters.

> 📘 Schmitz, T. B. *MST-Direct: Matching via Sinkhorn Transport for Multivariate
> Geostatistical Simulation with Complex Non-Linear Dependencies.*
> arXiv: [2603.18036](https://arxiv.org/abs/2603.18036)

> 📘 Schmitz, T. B. *MST-Direct at Scale: Multivariate and Conditional
> Geostatistical Simulation via Sinkhorn Optimal Transport.*
> arXiv: [2606.07578](https://arxiv.org/abs/2606.07578)

## What's new in 2.2

Version 2 scales the method to larger grids and adds the capabilities the
original (bivariate, unconditional, small-grid) formulation left open:

- **`ScalableMST`** — sparse, candidate-restricted Sinkhorn matcher, `O(n·C)`
  coupling storage. An independently documented run processed 40,000 nodes in
  about 40 seconds on a four-core Intel Core i7 870.
- **Multivariate** — handles many variables by matching the target cloud onto an
  FFT-MA Gaussian **backbone** with a prescribed variogram.
- **Conditional simulation** — honors hard data *exactly* by pinning the data
  tuples and conditioning the backbone by simple kriging.
- **PPMT** comparator, scalable variogram estimators, and a histogram-MSE metric.

## Install

```bash
pip install mst-direct            # core (numpy, scipy)
pip install 'mst-direct[plot]'    # + matplotlib for mst_direct.plots
```

## Quick start

```python
import numpy as np
from mst_direct import ScalableMST, gaussian_backbone, grid_coords, shape_preservation

N = 50
coords = grid_coords(N, N)                       # (2500, 2)
cloud  = np.random.default_rng(0).normal(size=(N * N, 3))   # target tuples (any joint)

backbone = gaussian_backbone((N, N), d=3, rng_range=15.0, random_state=0)
model = ScalableMST(n_candidates=48, max_candidates=96,
                    feature_transform="whiten", random_state=0)
sim = model.simulate(backbone, cloud, coords=coords)

# joint distribution preserved exactly (sim is a permutation of cloud)
assert np.array_equal(np.sort(sim, 0), np.sort(cloud, 0))
```

### Conditional simulation (honors hard data exactly)

```python
from mst_direct import conditional_gaussian_backbone, transport_features

data_idx   = np.array([0, 137, 999])             # hard-data grid locations
latent = transport_features(cloud, "whiten")
cond_bb = conditional_gaussian_backbone((N, N), coords, data_idx, latent[[5, 6, 7]],
                                        rng_range=15.0, random_state=0)
sim = ScalableMST(random_state=0).simulate(
    cond_bb, cloud, pinned=(data_idx, np.array([5, 6, 7])), coords=coords)
# sim[data_idx] == cloud[[5, 6, 7]] exactly
```

### v1 API (dense, bivariate) still available

```python
from mst_direct import MSTDirect, generate_dataset, shape_preservation
data = generate_dataset("gaussian_mix", grid=(25, 25), random_state=42)
sim = MSTDirect(random_state=42).simulate(data["values"], data["coords"])
print(shape_preservation(data["values"], sim))   # -> 1.0
```

## API

```
# simulators
MSTDirect(...)                       # v1 dense relational matcher
ScalableMST(beta, n_candidates, k_relational, lam_relational, n_relax, ...)
    .simulate(backbone, cloud, pinned=None, coords=None)
mst_simulate(backbone, cloud, ...)
transport_features(cloud, transform="whiten")

# backbone (prescribed variogram)
gaussian_backbone(shape, d, rng_range, random_state)
conditional_gaussian_backbone(shape, coords, data_idx, data_gauss, rng_range, ...)

# comparator
PPMT(n_iter, n_dirs, random_state).fit(x).forward(x) / .inverse(y)
ppmt_simulate(cloud, backbone, ...)

# optimal transport / spatial
sinkhorn, sinkhorn_plan, relational_match, greedy_round, knn_adjacency
spherical, exponential, gaussian, empirical_variogram, variogram_correlation
grid_coords, sampled_variogram, sampled_cross_variogram, fit_spherical

# metrics
shape_preservation, histogram2d_similarity, histogram_mse_table

# synthetic data
generate_dataset, make_grid, fft_ma, apply_relationship, RELATIONSHIPS

# plotting (optional: pip install mst-direct[plot])
mst_direct.plots: plot_scatter_matrix, plot_realization, plot_realization_grid,
                  plot_variograms, plot_data_honoring

# baselines (Gaussian Copula / LU) for comparison
mst_direct.baselines: gaussian_copula_simulate, lu_simulate
```

## Examples and reproducibility

`examples/basic_unconditional.py` and `examples/basic_conditional.py` are small,
tested usage examples. Run `python scripts/download_dms_data.py` to obtain the
MIT-licensed synthetic DMS benchmark inputs with checksum verification.

The real-data conditional hold-out workflow uses the public Swiss Jura
trace-metal dataset. Run `python scripts/download_jura_data.py`, install the
`real` extra, and execute `python scripts/jura_holdout_experiment.py`. Protocol,
results, and limitations are documented in `docs/JURA_HOLDOUT.md`.

The complete input/output contract, parameter guidance, diagnostics, expected
behaviour, and limitations are documented in `docs/USER_GUIDE.md`.

The historical scripts used to generate the two preprints' 200×200 and 100×100
figures were not included in the PyPI 2.0.0 or arXiv source archives. Version
2.2 replaces that unavailable workflow with
`scripts/reproduce_manuscript_figures.py`: a deterministic, end-to-end rerun
using the corrected matcher, checksum-verified DMS data, fixed seeds, and the
documented PPMT comparator. Publication figures (editable PDF and 360-dpi PNG),
machine-readable metrics, and compressed numerical results are stored under
`artifacts/manuscript_reproduction/`. See `docs/REPRODUCIBILITY.md` and
`VALIDATION.md`.

The initial independent scaling audit in `docs/SCALING_AUDIT.md` documents a
failure regime of the original raw-standardized configuration. Corrected
records in `corrected_screening_audit.json` and
`corrected_40k_validation.json` use latent whitening, adaptive candidate
support, nearest-free residual matching, and lag-stratified variograms.
Returned values remain exact permutations of the untouched cloud; matching
diagnostics are available in `model.diagnostics_`.

## License

MIT © 2026 **Tcharlies Bachmann Schmitz** — Data Science, PX.Center
