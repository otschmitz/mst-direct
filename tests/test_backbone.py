"""Tests for the FFT-MA Gaussian backbone (unconditional and conditional)."""

import numpy as np

from mst_direct import conditional_gaussian_backbone, gaussian_backbone, grid_coords


def test_unconditional_backbone_shape_and_stats():
    bb = gaussian_backbone((20, 20), d=3, rng_range=6.0, random_state=0)
    assert bb.shape == (400, 3)
    # each component is approximately standard normal
    assert np.allclose(bb.mean(axis=0), 0.0, atol=0.15)
    assert np.allclose(bb.std(axis=0), 1.0, atol=0.2)


def test_conditional_backbone_honors_gaussian_data():
    N = 20
    coords = grid_coords(N, N)
    rng = np.random.default_rng(0)
    data_idx = rng.choice(N * N, size=30, replace=False)
    data_gauss = rng.normal(size=(30, 3))
    bb = conditional_gaussian_backbone(
        (N, N), coords, data_idx, data_gauss, rng_range=6.0, random_state=0
    )
    assert bb.shape == (N * N, 3)
    # the conditioned field reproduces the Gaussian data at the data locations
    assert np.allclose(bb[data_idx], data_gauss)
