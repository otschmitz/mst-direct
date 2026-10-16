import numpy as np
import pytest

from mst_direct.adjacency import knn_adjacency, pairwise_distances, row_normalize
from mst_direct.synthetic import make_grid


def test_pairwise_distances_symmetric_zero_diagonal():
    coords = make_grid(4, 4)
    d = pairwise_distances(coords)
    np.testing.assert_allclose(d, d.T)
    np.testing.assert_allclose(np.diag(d), 0.0, atol=1e-12)


def test_knn_has_no_self_loops():
    coords = make_grid(5, 5)
    adj = knn_adjacency(coords, k=8)
    assert np.all(np.diag(adj) == 0.0)


def test_knn_symmetric_union():
    coords = make_grid(5, 5)
    adj = knn_adjacency(coords, k=4, symmetric=True)
    np.testing.assert_array_equal(adj, adj.T)


def test_knn_minimum_degree():
    # With symmetric union every point connects to at least k neighbors.
    coords = make_grid(6, 6)
    k = 8
    adj = knn_adjacency(coords, k=k)
    assert np.all(adj.sum(axis=1) >= k)


def test_knn_clamps_k_to_n_minus_one():
    coords = make_grid(2, 1)  # only 2 points
    adj = knn_adjacency(coords, k=8)
    assert adj.shape == (2, 2)
    assert adj[0, 1] == 1.0 and adj[1, 0] == 1.0


def test_knn_rejects_nonpositive_k():
    with pytest.raises(ValueError):
        knn_adjacency(make_grid(3, 3), k=0)


def test_row_normalize_rows_sum_to_one():
    adj = knn_adjacency(make_grid(4, 4), k=4)
    norm = row_normalize(adj)
    np.testing.assert_allclose(norm.sum(axis=1), 1.0, atol=1e-12)


def test_row_normalize_handles_isolated_rows():
    adj = np.zeros((3, 3))
    adj[0, 1] = 1.0
    norm = row_normalize(adj)
    assert np.all(np.isfinite(norm))
    assert norm[2].sum() == 0.0  # isolated row stays zero, not NaN
