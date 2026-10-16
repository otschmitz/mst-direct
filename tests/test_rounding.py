import numpy as np
import pytest

from mst_direct.rounding import greedy_round


def test_greedy_round_returns_permutation():
    rng = np.random.default_rng(0)
    m = rng.random((20, 20))
    perm = greedy_round(m)
    assert sorted(perm.tolist()) == list(range(20))


def test_greedy_round_identity_for_dominant_diagonal():
    n = 10
    m = np.eye(n) * 5.0 + 0.01
    perm = greedy_round(m)
    np.testing.assert_array_equal(perm, np.arange(n))


def test_greedy_round_resolves_conflicts_by_confidence():
    # Two rows both prefer column 0; the more confident one should win it.
    m = np.array(
        [
            [0.9, 0.1, 0.0],
            [0.8, 0.15, 0.05],
            [0.1, 0.1, 0.8],
        ]
    )
    perm = greedy_round(m)
    assert perm[0] == 0  # most confident on column 0
    assert sorted(perm.tolist()) == [0, 1, 2]


def test_greedy_round_requires_square():
    with pytest.raises(ValueError):
        greedy_round(np.ones((3, 4)))
