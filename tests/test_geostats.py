"""Tests for the scalable geostatistics utilities (variogram + spherical fit)."""

import numpy as np

from mst_direct import fit_spherical, grid_coords, sampled_variogram
from mst_direct.synthetic import fft_ma
from mst_direct.variograms import spherical


def test_sampled_variogram_rises_to_sill():
    field = fft_ma((40, 40), rng_range=10.0, random_state=0).ravel()
    coords = grid_coords(40, 40)
    cen, g, counts = sampled_variogram(field, coords, n_lags=12, max_dist=25, n_pairs=120_000)
    assert np.all(counts > 0)          # all lag bins populated
    assert g[0] < g[-1]                # variogram increases with lag
    assert np.isfinite(g).all()


def test_fit_spherical_recovers_range():
    lags = np.linspace(1.0, 30.0, 15)
    gamma = spherical(lags, sill=1.0, rng=12.0)
    rng_fit, sill_fit = fit_spherical(lags, gamma, sill=1.0)
    assert abs(rng_fit - 12.0) < 3.0
    assert abs(sill_fit - 1.0) < 0.2
