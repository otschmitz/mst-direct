"""Small executable unconditional MST-Direct example."""

import numpy as np

from mst_direct import ScalableMST, gaussian_backbone, grid_coords


def main() -> None:
    side = 30
    n = side * side
    rng = np.random.default_rng(7)
    x = rng.normal(size=n)
    cloud = np.column_stack((x, np.sin(x) + 0.2 * rng.normal(size=n)))
    coords = grid_coords(side, side)
    backbone = gaussian_backbone((side, side), 2, rng_range=10, random_state=7)
    model = ScalableMST(n_candidates=40, random_state=7)
    simulated = model.simulate(backbone, cloud, coords=coords)
    assert sorted(model.assignment_.tolist()) == list(range(n))
    assert np.array_equal(np.sort(simulated, axis=0), np.sort(cloud, axis=0))
    print(f"Generated {n} nodes; empirical tuples preserved exactly.")


if __name__ == "__main__":
    main()
