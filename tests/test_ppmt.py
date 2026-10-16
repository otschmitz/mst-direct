"""Tests for the PPMT comparator (forward/inverse transform and simulation)."""

import numpy as np

from mst_direct import PPMT, gaussian_backbone, ppmt_simulate


def test_invertibility(make_cloud):
    """inverse(forward(x)) must recover x to high precision."""
    x = make_cloud(800, seed=1)
    ppmt = PPMT(n_iter=20, n_dirs=40, random_state=1).fit(x)
    recon = ppmt.inverse(ppmt.forward(x))
    assert np.max(np.abs(recon - x)) < 1e-6


def test_gaussianization(make_cloud):
    """Forward factors are approximately unit-variance and decorrelated."""
    x = make_cloud(1000, seed=1)
    ppmt = PPMT(n_iter=25, n_dirs=50, random_state=1).fit(x)
    f = ppmt.forward(x)
    assert np.allclose(f.mean(axis=0), 0.0, atol=0.1)
    assert np.allclose(f.std(axis=0), 1.0, atol=0.25)
    cc = np.corrcoef(f, rowvar=False)
    off = cc[~np.eye(x.shape[1], dtype=bool)]
    assert np.max(np.abs(off)) < 0.2


def test_ppmt_simulate_shape_and_finite(make_cloud):
    N = 20
    cloud = make_cloud(N * N, seed=1)
    bb = gaussian_backbone((N, N), d=cloud.shape[1], rng_range=6.0, random_state=1)
    sim = ppmt_simulate(cloud, bb, n_iter=15, n_dirs=30, random_state=1)
    assert sim.shape == cloud.shape
    assert np.all(np.isfinite(sim))
