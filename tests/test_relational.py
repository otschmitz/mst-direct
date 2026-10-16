import numpy as np
import pytest

from mst_direct.adjacency import knn_adjacency
from mst_direct.relational import relational_match
from mst_direct.synthetic import make_grid


def test_relational_match_is_doubly_stochastic():
    coords = make_grid(8, 8)
    n = len(coords)
    rng = np.random.default_rng(0)
    s = rng.normal(size=(n, n))
    adj = knn_adjacency(coords, k=8)
    m = relational_match(s, adj, beta=35.0, lam=2.2, n_outer=10, n_inner=50)
    # Entropic Sinkhorn yields an approximately doubly-stochastic coupling.
    np.testing.assert_allclose(m.sum(axis=1), np.full(n, 1 / n), atol=1e-4)
    np.testing.assert_allclose(m.sum(axis=0), np.full(n, 1 / n), atol=1e-4)


def test_relational_match_nonnegative():
    coords = make_grid(6, 6)
    n = len(coords)
    s = np.eye(n)
    adj = knn_adjacency(coords, k=4)
    m = relational_match(s, adj, n_outer=5, n_inner=30)
    assert np.all(m >= 0.0)


def test_relational_term_changes_coupling():
    # lambda = 0 (no relational reward) should differ from lambda > 0.
    coords = make_grid(7, 7)
    n = len(coords)
    rng = np.random.default_rng(3)
    s = rng.normal(size=(n, n))
    adj = knn_adjacency(coords, k=8)
    m0 = relational_match(s, adj, beta=20.0, lam=0.0, n_outer=10, n_inner=50)
    m1 = relational_match(s, adj, beta=20.0, lam=3.0, n_outer=10, n_inner=50)
    assert not np.allclose(m0, m1)


def test_relational_match_converges_deterministically():
    coords = make_grid(6, 6)
    n = len(coords)
    rng = np.random.default_rng(5)
    s = rng.normal(size=(n, n))
    adj = knn_adjacency(coords, k=6)
    a = relational_match(s, adj, n_outer=15, n_inner=40)
    b = relational_match(s, adj, n_outer=15, n_inner=40)
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("lam", [0.0, 1.0, 2.2, 5.0])
def test_relational_match_stable_across_lambda(lam):
    coords = make_grid(6, 6)
    n = len(coords)
    s = np.cos(np.arange(n))[:, None] * np.cos(np.arange(n))[None, :]
    adj = knn_adjacency(coords, k=8)
    m = relational_match(s, adj, beta=35.0, lam=lam, n_outer=10, n_inner=40)
    assert np.all(np.isfinite(m))
