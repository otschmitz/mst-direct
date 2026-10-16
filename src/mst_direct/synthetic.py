"""Synthetic spatial data generation for the paper's experimental design.

Spatial Gaussian fields are produced with the **FFT Moving Average** (FFT-MA)
method (eq. 11)::

    Z(u) = F^{-1} { sqrt(S(omega)) * F{ W(u) } }

where ``F`` is the Fourier transform, ``S`` the spectral density of the target
covariance, and ``W`` white noise. On top of a base field ``X`` we impose one
of the five complex bivariate relationships of Table I to obtain ``Y``.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .variograms import exponential, spherical

__all__ = [
    "make_grid",
    "fft_ma",
    "RELATIONSHIPS",
    "apply_relationship",
    "generate_dataset",
]


def make_grid(nx: int, ny: int | None = None, spacing: float = 1.0) -> np.ndarray:
    """Return ``(nx * ny, 2)`` coordinates of a regular grid (row-major).

    The paper uses a ``25 x 25`` grid (625 points).
    """
    ny = nx if ny is None else ny
    ys, xs = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    coords = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float64)
    return coords * float(spacing)


def fft_ma(
    shape: tuple[int, int],
    model: Callable[..., np.ndarray] = spherical,
    rng_range: float = 12.0,
    sill: float = 1.0,
    random_state: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a standardized Gaussian random field via FFT-MA (eq. 11).

    Args:
        shape: ``(ny, nx)`` grid shape.
        model: A variogram model from :mod:`mst_direct.variograms`. The
            covariance is ``C(h) = sill - gamma(h)``.
        rng_range: Variogram range parameter ``a``.
        sill: Variogram sill.
        random_state: Seed or generator for the white-noise field.

    Returns:
        ``(ny, nx)`` field, standardized to zero mean and unit variance.
    """
    ny, nx = shape
    rng = np.random.default_rng(random_state)

    # Lag distances from the origin with periodic (toroidal) wrap-around.
    iy = np.minimum(np.arange(ny), ny - np.arange(ny))
    ix = np.minimum(np.arange(nx), nx - np.arange(nx))
    dy, dx = np.meshgrid(iy, ix, indexing="ij")
    h = np.sqrt(dy.astype(np.float64) ** 2 + dx.astype(np.float64) ** 2)

    cov = sill - model(h, sill=sill, rng=rng_range)

    spectrum = np.real(np.fft.fft2(cov))
    spectrum = np.clip(spectrum, 0.0, None)  # guard tiny negative round-off

    noise = rng.standard_normal((ny, nx))
    field = np.real(np.fft.ifft2(np.sqrt(spectrum) * np.fft.fft2(noise)))

    std = field.std()
    if std > 0:
        field = (field - field.mean()) / std
    return field


def _step(x, eps):
    return np.sign(x) * 0.8 + eps


def _gaussian_mix(x, eps, sign):
    return sign * x + eps


def _sinusoidal(x, eps):
    return np.sin(2.0 * np.pi * x) + eps


def _step_random(x, eps, sign):
    return sign * np.abs(x) + 0.25 * eps


def _heteroscedastic(x, eps):
    return x + np.abs(x) * eps


#: Names of the five complex bivariate relationships of Table I.
RELATIONSHIPS = (
    "step",
    "gaussian_mix",
    "sinusoidal",
    "step_random",
    "heteroscedastic",
)


def apply_relationship(
    x: np.ndarray,
    eps: np.ndarray,
    kind: str,
    random_state: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Compute ``Y`` from base field ``X`` and noise ``eps`` (Table I).

    Args:
        x: Base (spherical) field values, standardized.
        eps: Spatially correlated noise field, standardized.
        kind: One of :data:`RELATIONSHIPS`.
        random_state: Seed for the random sign in the branching relationships.

    Returns:
        The ``Y`` field, same shape as ``x``.
    """
    x = np.asarray(x, dtype=np.float64)
    eps = np.asarray(eps, dtype=np.float64)
    rng = np.random.default_rng(random_state)

    if kind == "step":
        return _step(x, eps)
    if kind == "gaussian_mix":
        sign = rng.choice([-1.0, 1.0], size=x.shape)
        return _gaussian_mix(x, eps, sign)
    if kind == "sinusoidal":
        return _sinusoidal(x, eps)
    if kind == "step_random":
        sign = rng.choice([-1.0, 1.0], size=x.shape)
        return _step_random(x, eps, sign)
    if kind == "heteroscedastic":
        return _heteroscedastic(x, eps)
    raise ValueError(f"unknown relationship {kind!r}; choose from {RELATIONSHIPS}")


def generate_dataset(
    kind: str,
    grid: tuple[int, int] = (25, 25),
    range_x: float = 12.0,
    range_y: float = 6.0,
    noise: float = 0.3,
    random_state: int | np.random.Generator | None = 42,
) -> dict[str, np.ndarray]:
    """Generate a synthetic bivariate spatial dataset (Section IV).

    Variable ``X`` is a spherical field (range 12); the noise driving ``Y`` is
    an independent exponential field (range 6), so ``Y`` carries its own
    spatial structure on top of its complex dependence on ``X``.

    Args:
        kind: One of :data:`RELATIONSHIPS`.
        grid: ``(nx, ny)`` grid size (paper: ``(25, 25)``).
        range_x: Spherical variogram range for ``X``.
        range_y: Exponential variogram range for the ``Y`` noise field.
        noise: Standard deviation scaling of the noise field ``eps``.
        random_state: Master seed (paper: 42).

    Returns:
        ``{"coords", "X", "Y", "values"}`` where ``coords`` is ``(n, 2)``,
        ``X``/``Y`` are ``(n,)`` and ``values`` is ``(n, 2)``.
    """
    nx, ny = grid
    seed = np.random.default_rng(random_state)
    # Independent seeds for the three random ingredients.
    s_x, s_eps, s_sign = seed.integers(0, 2**31 - 1, size=3)

    x_field = fft_ma((ny, nx), spherical, rng_range=range_x, random_state=int(s_x))
    eps_field = fft_ma((ny, nx), exponential, rng_range=range_y, random_state=int(s_eps))

    x = x_field.ravel()
    eps = noise * eps_field.ravel()
    y = apply_relationship(x, eps, kind, random_state=int(s_sign))

    coords = make_grid(nx, ny)
    values = np.column_stack([x, y])
    return {"coords": coords, "X": x, "Y": y, "values": values}
