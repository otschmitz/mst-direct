import numpy as np
import pytest

from mst_direct.metrics import histogram2d_similarity, shape_preservation


def test_identical_samples_similarity_one():
    rng = np.random.default_rng(0)
    sample = rng.normal(size=(200, 2))
    assert histogram2d_similarity(sample, sample, bins=20) == pytest.approx(1.0)


def test_permutation_preserves_shape_exactly():
    rng = np.random.default_rng(1)
    sample = rng.normal(size=(300, 2))
    permuted = sample[rng.permutation(300)]
    assert shape_preservation(sample, permuted, bins=15) == pytest.approx(1.0)


def test_disjoint_supports_similarity_zero():
    a = np.zeros((100, 2)) - 10.0
    b = np.zeros((100, 2)) + 10.0
    # Shared bin grid spans both; the populated bins never overlap.
    assert histogram2d_similarity(a, b, bins=20) == pytest.approx(0.0)


def test_similarity_in_unit_interval():
    rng = np.random.default_rng(2)
    a = rng.normal(size=(150, 2))
    b = rng.normal(size=(150, 2)) + 0.5
    s = histogram2d_similarity(a, b, bins=10)
    assert 0.0 <= s <= 1.0


def test_rejects_non_bivariate():
    with pytest.raises(ValueError):
        histogram2d_similarity(np.zeros((10, 3)), np.zeros((10, 3)))
