"""Tests for the scalable MST-Direct matcher (the paper's core contribution)."""

import numpy as np

from mst_direct import ScalableMST, gaussian_backbone, grid_coords, sampled_variogram
from mst_direct import shape_preservation


def test_exact_bijection_and_joint(make_cloud):
    """The simulation is a permutation of the cloud -> joint preserved exactly."""
    N = 24
    n = N * N
    cloud = make_cloud(n, seed=0)
    coords = grid_coords(N, N)
    bb = gaussian_backbone((N, N), d=cloud.shape[1], rng_range=8.0, random_state=1)

    mst = ScalableMST(beta=6.0, n_candidates=32, random_state=1)
    sim = mst.simulate(bb, cloud, coords=coords)

    # every cloud tuple used exactly once
    assert sorted(mst.assignment_.tolist()) == list(range(n))
    # output is the same multiset of tuples (exact joint distribution)
    assert np.array_equal(np.sort(sim, axis=0), np.sort(cloud, axis=0))
    # 2-D histogram similarity is exactly 1 on each variable pair
    for a, b in [(0, 1), (0, 2), (1, 2)]:
        assert shape_preservation(cloud[:, [a, b]], sim[:, [a, b]], bins=15) >= 0.999


def test_pinned_conditioning_honored_exactly(make_cloud):
    """Pinned (hard-data) locations output their assigned tuples exactly."""
    N = 24
    n = N * N
    cloud = make_cloud(n, seed=0)
    coords = grid_coords(N, N)
    bb = gaussian_backbone((N, N), d=cloud.shape[1], rng_range=8.0, random_state=2)

    rows = np.array([0, 5, 100, 300, 575])
    cols = np.array([10, 20, 30, 40, 50])
    mst = ScalableMST(beta=6.0, n_candidates=32, random_state=2)
    sim = mst.simulate(bb, cloud, pinned=(rows, cols), coords=coords)

    assert np.allclose(sim[rows], cloud[cols])             # data honored exactly
    assert sorted(mst.assignment_.tolist()) == list(range(n))  # still a bijection


def test_variogram_reproduction(make_cloud):
    """The simulated field inherits the backbone's spatial correlation."""
    N = 30
    n = N * N
    cloud = make_cloud(n, seed=4)
    coords = grid_coords(N, N)
    rng_range = 10.0
    bb = gaussian_backbone((N, N), d=cloud.shape[1], rng_range=rng_range, random_state=4)

    sim = ScalableMST(beta=8.0, n_candidates=40, lam_relational=0.5, n_relax=2,
                      random_state=4).simulate(bb, cloud, coords=coords)

    _, g_bb, _ = sampled_variogram(bb[:, 0], coords, max_dist=25, n_pairs=80_000)
    _, g_sim, _ = sampled_variogram(sim[:, 0], coords, max_dist=25, n_pairs=80_000)
    m = np.isfinite(g_bb) & np.isfinite(g_sim)
    corr = np.corrcoef(g_bb[m], g_sim[m])[0, 1]
    assert corr > 0.8
