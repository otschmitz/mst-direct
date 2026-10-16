"""FFT-MA Gaussian backbone fields carrying the prescribed spatial structure.

DMS draws ``d`` *independent, spatially correlated Gaussian realizations* with
a chosen variogram and then transforms them to the target distribution. We do
the same to obtain the spatial template that MST-Direct's Sinkhorn matching
maps the reference cloud onto. The Gaussian components are independent (no
cross-correlation), exactly as in DMS Section 4.

Unconditional backbones come straight from FFT-MA
(:func:`mst_direct.synthetic.fft_ma`). Conditional backbones honor hard data
in the Gaussian domain via the classic *conditioning by simple kriging*
(Journel & Huijbregts, 1978):

    Z_c(u) = Z_u(u) + [ SK(z_d)(u) - SK(Z_u(d))(u) ]

so that ``Z_c`` interpolates the data while keeping the unconditional
covariance.
"""

from __future__ import annotations

import numpy as np

from .synthetic import fft_ma
from .variograms import spherical

__all__ = ["gaussian_backbone", "conditional_gaussian_backbone"]


def gaussian_backbone(
    shape: tuple[int, int],
    d: int,
    rng_range: float,
    random_state: int | None = None,
) -> np.ndarray:
    """``(nx*ny, d)`` independent standardized Gaussian fields (spherical model).

    Args:
        shape: ``(ny, nx)`` grid shape.
        d: Number of independent components (6 for the DMS experiments).
        rng_range: Spherical variogram range (DMS: 40 uncond, 18 cond).
        random_state: Master seed; each component uses a derived sub-seed.

    Returns:
        ``(ny*nx, d)`` array (row-major), each column N(0, 1) with the target
        spatial correlation.
    """
    ny, nx = shape
    seeder = np.random.default_rng(random_state)
    seeds = seeder.integers(0, 2**31 - 1, size=d)
    cols = [
        fft_ma((ny, nx), spherical, rng_range=rng_range, random_state=int(s)).ravel()
        for s in seeds
    ]
    return np.column_stack(cols)


def _spherical_cov(h: np.ndarray, rng_range: float, sill: float = 1.0) -> np.ndarray:
    """Spherical covariance ``C(h) = sill - gamma(h)``."""
    return sill - spherical(h, sill=sill, rng=rng_range)


def _simple_kriging_operator(
    coords: np.ndarray,
    data_coords: np.ndarray,
    rng_range: float,
    nugget: float = 1e-6,
) -> np.ndarray:
    """Return the ``(n_grid, n_data)`` simple-kriging weight operator ``W``.

    ``SK estimate at grid = W @ data_residuals`` (zero-mean simple kriging).
    """
    nd = len(data_coords)
    dd = np.hypot(
        data_coords[:, 0][:, None] - data_coords[:, 0][None, :],
        data_coords[:, 1][:, None] - data_coords[:, 1][None, :],
    )
    Cdd = _spherical_cov(dd, rng_range)
    Cdd[np.diag_indices(nd)] += nugget
    gd = np.hypot(
        coords[:, 0][:, None] - data_coords[:, 0][None, :],
        coords[:, 1][:, None] - data_coords[:, 1][None, :],
    )
    Cgd = _spherical_cov(gd, rng_range)
    # W = Cgd @ inv(Cdd)
    return Cgd @ np.linalg.inv(Cdd)


def conditional_gaussian_backbone(
    shape: tuple[int, int],
    coords: np.ndarray,
    data_idx: np.ndarray,
    data_gauss: np.ndarray,
    rng_range: float,
    random_state: int | None = None,
) -> np.ndarray:
    """Conditional Gaussian backbone honoring Gaussian-domain hard data.

    Args:
        shape: ``(ny, nx)`` grid shape.
        coords: ``(n, 2)`` grid coordinates (row-major, matching ``shape``).
        data_idx: ``(m,)`` flat grid indices of the conditioning locations.
        data_gauss: ``(m, d)`` Gaussian-domain hard-data values.
        rng_range: Spherical variogram range (DMS conditional: ~18).
        random_state: Master seed.

    Returns:
        ``(n, d)`` conditional Gaussian fields that interpolate ``data_gauss``
        at ``data_idx`` and follow the spherical covariance elsewhere.
    """
    coords = np.asarray(coords, dtype=np.float64)
    data_idx = np.asarray(data_idx, dtype=np.int64)
    data_gauss = np.asarray(data_gauss, dtype=np.float64)
    d = data_gauss.shape[1]

    uncond = gaussian_backbone(shape, d, rng_range, random_state=random_state)
    data_coords = coords[data_idx]
    W = _simple_kriging_operator(coords, data_coords, rng_range)

    cond = np.empty_like(uncond)
    for c in range(d):
        sk_data = W @ data_gauss[:, c]               # kriging of the hard data
        sk_unc = W @ uncond[data_idx, c]             # kriging of the uncond field at data
        cond[:, c] = uncond[:, c] + sk_data - sk_unc
    # Enforce exact data reproduction (kriging is exact up to the nugget).
    cond[data_idx, :] = data_gauss
    return cond
