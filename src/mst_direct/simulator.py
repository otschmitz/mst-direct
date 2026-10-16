"""High-level MST-Direct multivariate geostatistical simulator (Algorithm 1).

The :class:`MSTDirect` class ties together adjacency construction, relational
Sinkhorn matching, and greedy rounding into a single estimator-style
interface::

    >>> import numpy as np
    >>> from mst_direct import MSTDirect, make_grid
    >>> coords = make_grid(10, 10)
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(size=len(coords))
    >>> y = np.sign(x) * 0.8 + 0.1 * rng.normal(size=len(coords))
    >>> sim = MSTDirect(k=8, random_state=42).simulate(np.column_stack([x, y]), coords)
    >>> sim.shape
    (100, 2)

Because the anchor set ``U`` is a random permutation of the (normalized) input
(steps 5-6) and greedy rounding yields a bijection (step 9), the simulated
field is a *spatial rearrangement* of the original value tuples. Its joint
distribution is therefore identical to the original (100% shape preservation
by construction), while the relational matching relocates the tuples to
reproduce the spatial variogram structure.
"""

from __future__ import annotations

import numpy as np

from .adjacency import knn_adjacency
from .relational import relational_match
from .rounding import greedy_round

__all__ = ["MSTDirect"]


def _unit_rows(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize each row onto the unit sphere (Algorithm 1, steps 4 & 6)."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > eps)


class MSTDirect:
    """Matching via Sinkhorn Transport — direct multivariate simulation.

    Args:
        k: Number of nearest neighbors for the spatial adjacency graph.
            Paper: 8.
        beta: Inverse Sinkhorn regularization (matching sharpness). Paper: 35.0.
        lam: Relational weight (spatial vs. feature consistency). Paper: 2.2.
        n_outer: Maximum relational refinement iterations.
        n_inner: Sinkhorn iterations per refinement.
        tol: Convergence tolerance.
        random_state: Seed (or ``np.random.Generator``) for anchor generation.
            Fix it for reproducibility (the paper uses 42).

    Attributes (populated after :meth:`simulate`):
        coupling_: The soft coupling matrix ``M*`` of shape ``(n, n)``.
        assignment_: The hard permutation ``pi`` (length ``n``).
        anchor_permutation_: The random permutation used to build the anchors.
        mean_, std_: Per-variable normalization statistics.
    """

    def __init__(
        self,
        k: int = 8,
        beta: float = 35.0,
        lam: float = 2.2,
        n_outer: int = 20,
        n_inner: int = 100,
        tol: float = 1e-7,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        if k <= 0:
            raise ValueError("k must be a positive integer")
        self.k = int(k)
        self.beta = float(beta)
        self.lam = float(lam)
        self.n_outer = int(n_outer)
        self.n_inner = int(n_inner)
        self.tol = float(tol)
        self.random_state = random_state

        self.coupling_: np.ndarray | None = None
        self.assignment_: np.ndarray | None = None
        self.anchor_permutation_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def simulate(self, values: np.ndarray, coords: np.ndarray) -> np.ndarray:
        """Run the full MST-Direct pipeline (Algorithm 1, steps 1-10).

        Args:
            values: ``(n, d)`` array stacking the ``d`` variables to simulate
                jointly (e.g. ``np.column_stack([x, y])``). A list of ``d``
                1-D arrays is also accepted.
            coords: ``(n, 2)`` spatial coordinates of the ``n`` locations.

        Returns:
            ``(n, d)`` simulated field — a spatial rearrangement of ``values``.
        """
        v = np.asarray(values, dtype=np.float64)
        if v.ndim == 1:
            v = v[:, None]
        coords = np.asarray(coords, dtype=np.float64)
        n = v.shape[0]
        if coords.shape[0] != n:
            raise ValueError(
                f"values has {n} rows but coords has {coords.shape[0]}"
            )
        if n < 2:
            raise ValueError("need at least 2 points to simulate")

        # Steps 2-3: normalize to zero mean / unit variance (per variable).
        mu = v.mean(axis=0)
        sigma = v.std(axis=0)
        sigma_safe = np.where(sigma > 0, sigma, 1.0)
        v_norm = (v - mu) / sigma_safe

        # Step 4: project data points onto the unit sphere (cosine matching).
        v_unit = _unit_rows(v_norm)

        # Steps 5-6: anchors are a random permutation of the normalized data,
        # so they share the original marginals exactly.
        rng = np.random.default_rng(self.random_state)
        anchor_perm = rng.permutation(n)
        u = v_norm[anchor_perm]
        u_unit = _unit_rows(u)

        # Step 7: spatial k-NN adjacency.
        adj = knn_adjacency(coords, self.k)

        # Step 8: relational Sinkhorn matching.
        similarity = v_unit @ u_unit.T
        coupling = relational_match(
            similarity,
            adj,
            beta=self.beta,
            lam=self.lam,
            n_outer=self.n_outer,
            n_inner=self.n_inner,
            tol=self.tol,
        )

        # Step 9: greedy rounding to a hard permutation.
        pi = greedy_round(coupling)

        # Step 10: gather matched anchors and denormalize back to data scale.
        result_norm = u[pi]
        result = result_norm * sigma_safe + mu

        self.coupling_ = coupling
        self.assignment_ = pi
        self.anchor_permutation_ = anchor_perm
        self.mean_ = mu
        self.std_ = sigma

        return result

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"MSTDirect(k={self.k}, beta={self.beta}, lam={self.lam}, "
            f"random_state={self.random_state!r})"
        )
