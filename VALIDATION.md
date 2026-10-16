# Validation record

Validation date: 2026-10-16 (America/Sao_Paulo).

Environment: Linux, Python 3.12.3. Exact package versions are listed in
`requirements-lock.txt`.

Commands and results:

```text
pytest -q
65 passed

python examples/basic_unconditional.py
Generated 900 nodes; empirical tuples preserved exactly.

python examples/basic_conditional.py
Generated 576 nodes; 4 hard-data tuples honoured.

python scripts/download_dms_data.py
Both files downloaded and SHA-256 checksums matched.
```

This validates the packaged unit tests and small examples. It does not validate
the numerical results printed in the manuscript because the original
manuscript experiment scripts were not released.

An additional independent audit rerun is implemented in
`scripts/independent_audit_benchmark.py`, with output captured in
`audit_results.json`. On an Intel Core i7 870 (4 cores/8 threads, 15 GiB RAM),
the default non-relational matcher completed the 40,000-node match in 38.83 s
and the 10,000-node conditional match in 6.70 s. Process peak RSS was 403,568
KiB. Both assignments were permutations; the conditional run honoured all 200
data tuples with maximum absolute error zero. These results support the narrow
runtime and invariant claims for this independent configuration, but do not
reproduce spatial-quality figures or the undisclosed relational configuration.
