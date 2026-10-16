# Contributing

Bug reports and pull requests are welcome. Please include a minimal reproducible
example and report Python, NumPy, and SciPy versions.

Before opening a pull request, run:

```bash
python -m pip install -e '.[dev]'
pytest -q
```

Scientific changes should add tests, state assumptions, and report failure
regimes. Do not add private datasets, credentials, generated environments, or
third-party material without a compatible licence.
