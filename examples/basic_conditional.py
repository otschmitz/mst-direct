"""Small executable conditional MST-Direct example with pinned tuples."""

import numpy as np

from mst_direct import (
    ScalableMST,
    conditional_gaussian_backbone,
    grid_coords,
    transport_features,
)


def main() -> None:
    side = 24
    n = side * side
    rng = np.random.default_rng(11)
    x = rng.normal(size=n)
    cloud = np.column_stack((x, x**2 + 0.15 * rng.normal(size=n)))
    coords = grid_coords(side, side)
    data_rows = np.array([0, 100, 250, 575])
    data_cloud_rows = np.array([3, 9, 21, 37])
    latent = transport_features(cloud, "whiten")
    backbone = conditional_gaussian_backbone(
        (side, side), coords, data_rows, latent[data_cloud_rows],
        rng_range=8, random_state=11,
    )
    model = ScalableMST(n_candidates=32, random_state=11)
    simulated = model.simulate(
        backbone, cloud, pinned=(data_rows, data_cloud_rows), coords=coords
    )
    assert np.array_equal(simulated[data_rows], cloud[data_cloud_rows])
    print(f"Generated {n} nodes; {len(data_rows)} hard-data tuples honoured.")


if __name__ == "__main__":
    main()
