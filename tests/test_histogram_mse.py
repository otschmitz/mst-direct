"""Tests for the v2 histogram-MSE reproduction metric."""

import numpy as np

from mst_direct import histogram_mse_table


def test_histogram_mse_zero_for_identical(make_cloud):
    x = make_cloud(500, seed=0)
    table = histogram_mse_table(x, x, bins=20)
    assert table.shape == (3, 3)
    assert np.allclose(table, 0.0)


def test_histogram_mse_positive_for_different(make_cloud):
    x = make_cloud(500, seed=0)
    y = make_cloud(500, seed=1)
    table = histogram_mse_table(x, y, bins=20)
    assert table.mean() > 0.0
