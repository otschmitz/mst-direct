import numpy as np
import pytest

from mst_direct import generate_dataset, shape_preservation

pytest.importorskip("scipy")

from mst_direct.baselines import gaussian_copula_simulate, lu_simulate  # noqa: E402


def test_copula_preserves_marginals():
    data = generate_dataset("gaussian_mix", grid=(12, 12), random_state=42)
    sim = gaussian_copula_simulate(data["values"], data["coords"], random_state=0)
    # Rank back-transform reproduces each marginal exactly (as a multiset).
    for c in range(2):
        np.testing.assert_allclose(
            np.sort(sim[:, c]), np.sort(data["values"][:, c]), atol=1e-9
        )


def test_lu_preserves_marginals():
    data = generate_dataset("step", grid=(12, 12), random_state=42)
    sim = lu_simulate(data["values"], data["coords"], random_state=0)
    for c in range(2):
        np.testing.assert_allclose(
            np.sort(sim[:, c]), np.sort(data["values"][:, c]), atol=1e-9
        )


def test_baselines_degrade_shape_on_nonlinear_relationship():
    # The whole point of the paper: Gaussian baselines cannot reproduce a
    # bimodal joint, so their shape preservation is well below 1.0.
    data = generate_dataset("gaussian_mix", grid=(15, 15), random_state=42)
    cop = gaussian_copula_simulate(data["values"], data["coords"], random_state=0)
    lu = lu_simulate(data["values"], data["coords"], random_state=0)
    assert shape_preservation(data["values"], cop) < 0.9
    assert shape_preservation(data["values"], lu) < 0.9


def test_baselines_output_shape():
    data = generate_dataset("sinusoidal", grid=(10, 10), random_state=1)
    assert gaussian_copula_simulate(data["values"], data["coords"]).shape == (100, 2)
    assert lu_simulate(data["values"], data["coords"]).shape == (100, 2)
