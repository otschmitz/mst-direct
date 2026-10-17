# Reproducibility status

The repository tree was reconstructed from the public PyPI 2.0.0 source
distribution. It repairs the omitted shared test fixture and provides small
executable examples plus a checksum-verified DMS data downloader.

The exact historical scripts and complete parameter record used for the two
preprints were absent from the PyPI and arXiv source archives. They could not be
recovered from the Git history, branches, releases, tags, or public gists.

Version 2.2 therefore supplies an independent, deterministic replacement in
`scripts/reproduce_manuscript_figures.py`. It regenerates every synthetic DMS
figure and histogram table used by the journal manuscript with the corrected
implementation. Outputs are committed under `artifacts/manuscript_reproduction`:

- editable PDF and 360-dpi PNG figures;
- `manuscript_reproduction.json` with configuration, environment, diagnostics,
  timings, histogram tables, and variogram values; and
- compressed arrays for the unconditional and conditional experiments.

Run from a locked environment with:

```text
python scripts/download_dms_data.py
python scripts/reproduce_manuscript_figures.py
```

These are explicitly described as a corrected independent reproduction, not
as the unavailable historical workflow.

Required environment: Python 3.9 or newer, NumPy, SciPy; Matplotlib for plots
and pytest for tests. The committed JSON records exact library and operating
system versions plus wall times. Peak RSS for the separate scaling audit is
recorded in `audit_results.json`.
