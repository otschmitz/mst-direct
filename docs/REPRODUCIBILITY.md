# Reproducibility status

The repository tree was reconstructed from the public PyPI 2.0.0 source
distribution. It repairs the omitted shared test fixture and provides small
executable examples plus a checksum-verified DMS data downloader.

It does **not** yet reproduce the manuscript's numerical tables and figures:
the exact scripts and complete parameter record used to create those outputs
were absent from the PyPI and arXiv source archives. Any newly written benchmark
script must be described as an independent rerun, not the original experiment.

Required environment: Python 3.9 or newer, NumPy, SciPy; Matplotlib for plots
and pytest for tests. Record exact versions, operating system, CPU, wall time,
and peak resident memory for every manuscript benchmark.
