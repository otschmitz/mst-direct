import numpy as np
import pytest

from mst_direct.synthetic import make_grid
from mst_direct.variograms import (
    empirical_variogram,
    exponential,
    spherical,
    variogram_correlation,
)


def test_spherical_reaches_sill_at_range():
    assert spherical(np.array([1.0]), sill=2.0, rng=1.0)[0] == pytest.approx(2.0)
    assert spherical(np.array([5.0]), sill=2.0, rng=1.0)[0] == pytest.approx(2.0)


def test_spherical_zero_at_origin():
    assert spherical(np.array([0.0]), sill=1.0, rng=10.0)[0] == 0.0


def test_spherical_monotonic_increasing():
    h = np.linspace(0, 12, 50)
    g = spherical(h, sill=1.0, rng=12.0)
    assert np.all(np.diff(g) >= -1e-12)


def test_exponential_practical_range():
    # At h = range the exponential model is ~95% of the sill.
    g = exponential(np.array([6.0]), sill=1.0, rng=6.0)[0]
    assert g == pytest.approx(1.0 - np.exp(-3.0), abs=1e-9)


def test_empirical_variogram_increases_for_smooth_field():
    coords = make_grid(15, 15)
    # A smooth linear trend -> semivariance grows with lag.
    values = coords[:, 0].astype(float)
    centers, gamma, counts = empirical_variogram(values, coords, n_lags=10)
    finite = gamma[np.isfinite(gamma)]
    assert finite[0] < finite[-1]
    assert counts.sum() > 0


def test_variogram_correlation_perfect_for_identical():
    g = np.array([0.1, 0.4, 0.7, 0.9, 1.0])
    assert variogram_correlation(g, g) == pytest.approx(1.0)


def test_variogram_correlation_drops_nan_bins():
    a = np.array([0.1, np.nan, 0.7, 0.9])
    b = np.array([0.2, 0.5, 0.8, 1.0])
    r = variogram_correlation(a, b)
    assert np.isfinite(r)
