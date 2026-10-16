import numpy as np
import pytest

from mst_direct.sinkhorn import logsumexp, sinkhorn, sinkhorn_plan


def test_logsumexp_matches_naive():
    a = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]])
    expected = np.log(np.exp(a).sum(axis=1, keepdims=True))
    np.testing.assert_allclose(logsumexp(a, axis=1), expected)


def test_logsumexp_stable_for_large_values():
    a = np.array([[1000.0, 1001.0]])
    # Naive exp would overflow; the stable version must not.
    val = logsumexp(a, axis=1)
    assert np.isfinite(val).all()


def test_sinkhorn_marginals_are_uniform():
    rng = np.random.default_rng(0)
    cost = rng.random((12, 12))
    m = sinkhorn(cost, beta=10.0, n_iter=500)
    np.testing.assert_allclose(m.sum(axis=1), np.full(12, 1 / 12), atol=1e-6)
    np.testing.assert_allclose(m.sum(axis=0), np.full(12, 1 / 12), atol=1e-6)


def test_sinkhorn_total_mass_is_one():
    rng = np.random.default_rng(1)
    cost = rng.random((8, 8))
    m = sinkhorn(cost, beta=20.0)
    assert m.sum() == pytest.approx(1.0, abs=1e-6)


def test_sinkhorn_prefers_low_cost():
    # A near-diagonal cost should concentrate transport on the diagonal.
    n = 6
    cost = np.ones((n, n)) - np.eye(n)
    m = sinkhorn(cost, beta=50.0, n_iter=1000)
    assert np.trace(m) > m.sum() - np.trace(m)  # diagonal carries most mass


def test_sinkhorn_plan_custom_marginals():
    log_kernel = np.zeros((3, 3))  # uniform kernel
    a = np.array([0.5, 0.3, 0.2])
    b = np.array([0.2, 0.3, 0.5])
    m = sinkhorn_plan(log_kernel, np.log(a), np.log(b), n_iter=500)
    np.testing.assert_allclose(m.sum(axis=1), a, atol=1e-6)
    np.testing.assert_allclose(m.sum(axis=0), b, atol=1e-6)
