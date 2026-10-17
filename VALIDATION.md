# Validation record

Validation date: 2026-10-16 (America/Sao_Paulo).

Environment: Linux, Python 3.12.3. Exact package versions are listed in
`requirements-lock.txt`.

Commands and results:

```text
pytest -q
67 passed

python examples/basic_unconditional.py
Generated 900 nodes; empirical tuples preserved exactly.

python examples/basic_conditional.py
Generated 576 nodes; 4 hard-data tuples honoured.

python scripts/download_dms_data.py
Both files downloaded and SHA-256 checksums matched.
```

This validates the packaged unit tests and small examples. The corrected
manuscript workflow is additionally validated by:

```text
python scripts/reproduce_manuscript_figures.py
```

The run completed on 17 October 2026 and produced complete permutations for
both DMS experiments. The 40,000-node match took 65.68 s with a 4.99% repair
fraction. The 10,000-node conditional match took 13.57 s, honoured all 200 hard
tuples exactly, and had a 1.68% repair fraction. The PPMT conditioning error was
1.85e-13. Full values and figures are under
`artifacts/manuscript_reproduction/`.

An additional independent audit rerun is implemented in
`scripts/independent_audit_benchmark.py`, with output captured in
`audit_results.json`. On an Intel Core i7 870 (4 cores/8 threads, 15 GiB RAM),
the default non-relational matcher completed the 40,000-node match in 38.83 s
and the 10,000-node conditional match in 6.70 s. Process peak RSS was 403,568
KiB. Both assignments were permutations; the conditional run honoured all 200
data tuples with maximum absolute error zero. These results support the narrow
runtime and invariant claims for this independent configuration, but do not
reproduce spatial-quality figures or the undisclosed relational configuration.

## Corrected matcher screening

`scripts/corrected_screening_audit.py` evaluates three 10,000-node simulations
with independent evaluation seeds and 10,000 sampled pairs per lag bin. The
minimum per-variable variogram-shape correlations were 0.901--0.971, maximum
normalized shape RMSE was 0.238, and residual repair affected 2.91--3.62% of
rows. All outputs were exact tuple permutations. Full records are in
`corrected_screening_audit.json`.

`scripts/corrected_40k_validation.py` then evaluated the 40,000-node seed-1
case. Matching took 65.98 s with 96 adaptive candidates; 4.99% of rows entered
nearest-free residual repair. Across three independent metric samples, minimum
variogram correlations were 0.948--0.959, maximum normalized shape RMSE was
0.229, and estimated sills were 0.997--1.096 times their target cloud
variances. See `corrected_40k_validation.json`.
