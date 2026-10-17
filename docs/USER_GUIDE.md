# MST-Direct user guide

## Inputs and outputs

`ScalableMST.simulate` accepts an `(n, d)` Gaussian backbone and an `(n, d)`
target tuple cloud. It returns an `(n, d)` field that is a row permutation of
the cloud. The row count must match because the empirical-distribution
guarantee is bijective. Coordinates are required only for relational
refinement. Conditional runs additionally receive equal-length arrays of grid
rows and cloud rows to pin.

## Recommended workflow

1. Build or load a representative multivariate tuple cloud.
2. Create independent Gaussian backbone components with the desired spatial
   covariance using `gaussian_backbone`, or condition them with
   `conditional_gaussian_backbone`.
3. Run `ScalableMST` with whitening, adaptive candidate support, and fixed
   random seeds for reproducible studies.
4. Inspect `model.diagnostics_`. Always report effective candidate count,
   zero-degree columns, greedy unmatched fraction, repair rounds, and final
   match distances.
5. Validate distributional invariants separately from spatial quality. Exact
   histogram preservation is guaranteed by permutation and is not a predictive
   score.

## Main options

- `beta`: inverse entropic regularization; larger values sharpen the coupling.
- `n_candidates`, `max_candidates`: initial and maximum sparse support sizes.
- `candidate_zero_tolerance`: threshold that triggers support expansion.
- `feature_transform`: use `whiten` for dependent clouds matched to independent
  Gaussian backbone components.
- `n_iter`, `tol`: Sinkhorn stopping controls.
- `lam_relational`, `n_relax`: optional neighbour-mean refinement. Keep these
  at zero unless their sensitivity is evaluated and reported.

## Expected behaviour and limitations

The returned field preserves complete supplied tuples exactly. Spatial
covariance reproduction is approximate. Adaptive support checks a necessary
degree condition but does not prove that the sparse graph has a perfect
matching. Residual rows are therefore assigned by nearest-free repair; a large
repair fraction is a diagnostic warning. Nearest-neighbour performance can
also deteriorate as dimension grows.

## Executable examples

- `examples/basic_unconditional.py`: small unconditional simulation.
- `examples/basic_conditional.py`: exact tuple pinning.
- `scripts/reproduce_manuscript_figures.py`: full DMS manuscript workflow.
- `scripts/jura_holdout_experiment.py`: Swiss Jura blind hold-out workflow.

See `docs/REPRODUCIBILITY.md` for exact commands and `VALIDATION.md` for the
recorded validation status.
