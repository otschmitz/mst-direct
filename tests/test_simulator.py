import numpy as np
import pytest

from mst_direct import MSTDirect, generate_dataset, shape_preservation
from mst_direct.synthetic import make_grid


def _is_row_permutation(a, b, decimals=9):
    """True iff ``b`` is a row permutation of ``a`` (order-independent)."""
    sa = np.array(sorted(map(tuple, np.round(a, decimals))))
    sb = np.array(sorted(map(tuple, np.round(b, decimals))))
    return np.allclose(sa, sb)


def test_output_is_permutation_of_input():
    data = generate_dataset("step", grid=(15, 15), random_state=42)
    sim = MSTDirect(random_state=42).simulate(data["values"], data["coords"])
    assert sim.shape == data["values"].shape
    assert _is_row_permutation(data["values"], sim)


def test_assignment_is_a_bijection():
    data = generate_dataset("sinusoidal", grid=(12, 12), random_state=0)
    model = MSTDirect(random_state=1)
    model.simulate(data["values"], data["coords"])
    assert sorted(model.assignment_.tolist()) == list(range(len(data["coords"])))


@pytest.mark.parametrize(
    "kind", ["step", "gaussian_mix", "sinusoidal", "step_random", "heteroscedastic"]
)
def test_shape_preservation_is_exactly_one(kind):
    # Because the output is a permutation of the input, the joint histogram is
    # identical -> the paper's 100% shape preservation holds by construction.
    data = generate_dataset(kind, grid=(15, 15), random_state=42)
    sim = MSTDirect(random_state=42).simulate(data["values"], data["coords"])
    assert shape_preservation(data["values"], sim) == pytest.approx(1.0, abs=1e-12)


def test_determinism_with_fixed_seed():
    data = generate_dataset("gaussian_mix", grid=(12, 12), random_state=7)
    a = MSTDirect(random_state=123).simulate(data["values"], data["coords"])
    b = MSTDirect(random_state=123).simulate(data["values"], data["coords"])
    np.testing.assert_array_equal(a, b)


def test_accepts_separate_1d_arrays():
    coords = make_grid(10, 10)
    rng = np.random.default_rng(0)
    x = rng.normal(size=len(coords))
    y = np.sign(x) * 0.8 + 0.1 * rng.normal(size=len(coords))
    sim = MSTDirect(random_state=0).simulate(np.column_stack([x, y]), coords)
    assert sim.shape == (100, 2)


def test_supports_more_than_two_variables():
    coords = make_grid(8, 8)
    rng = np.random.default_rng(2)
    v = rng.normal(size=(len(coords), 3))
    sim = MSTDirect(random_state=0).simulate(v, coords)
    assert sim.shape == (64, 3)
    assert _is_row_permutation(v, sim)


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        MSTDirect().simulate(np.zeros((10, 2)), make_grid(3, 3))
