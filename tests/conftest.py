"""Shared deterministic fixtures omitted from the PyPI 2.0.0 source archive."""

import numpy as np
import pytest


@pytest.fixture
def make_cloud():
    """Return a generator for a nonlinear, heteroscedastic three-variate cloud."""

    def _make(n: int, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        x = rng.normal(size=n)
        y = np.sin(1.5 * x) + (0.20 + 0.15 * np.abs(x)) * rng.normal(size=n)
        z = np.where(x < 0, -1.0, 1.0) + 0.30 * y + 0.15 * rng.normal(size=n)
        return np.column_stack((x, y, z))

    return _make
