"""Relational Sinkhorn matching (Algorithm 1, step 8; equations 7-8).

The plain Sinkhorn algorithm matches two point sets by *feature* similarity
alone, which would relocate values globally without respecting local spatial
relationships. MST-Direct augments the cost with a **relational** term so that
if point ``i`` is a spatial neighbor of point ``k`` in the source, their
matched anchors should also be neighbors in the target:

    C_ij = -v_i . u_j  -  lambda * sum_{k,l} A_V(i,k) M_kl A_U(j,l)        (eq. 7)

which gives the combined objective (the Gibbs kernel ``K = exp(-beta C)``)

    M_ij  propto  exp( beta * ( v_i . u_j  +  lambda * (A_V M A_U^T)_ij ) ) (eq. 8)

This is solved by alternating between (a) recomputing the relational reward
``R = A_V M A_U^T`` from the current coupling and (b) re-running Sinkhorn
normalization (eq. 6). The relational term is propagated with row-stochastic
adjacency matrices and a row-normalized coupling so that ``R`` stays on the
same ``[0, 1]`` scale as the cosine feature similarity, keeping ``beta`` and
``lambda`` numerically well-behaved.
"""

from __future__ import annotations

import numpy as np

from .adjacency import row_normalize
from .sinkhorn import sinkhorn_plan

__all__ = ["relational_match"]


def relational_match(
    similarity: np.ndarray,
    adjacency: np.ndarray,
    beta: float = 35.0,
    lam: float = 2.2,
    adjacency_anchor: np.ndarray | None = None,
    n_outer: int = 20,
    n_inner: int = 100,
    tol: float = 1e-7,
) -> np.ndarray:
    """Compute the relational Sinkhorn coupling ``M*`` (eq. 7-8).

    Args:
        similarity: ``(n, n)`` feature-similarity matrix ``S_ij = v_i . u_j``
            (cosine similarity of unit feature vectors, so ``S in [-1, 1]``).
        adjacency: ``(n, n)`` source spatial adjacency ``A_V`` (see
            :func:`mst_direct.adjacency.knn_adjacency`).
        beta: Inverse Sinkhorn regularization (kernel sharpness). Paper: 35.0.
        lam: Relational weight balancing feature vs. spatial consistency.
            Paper: 2.2.
        adjacency_anchor: Optional target adjacency ``A_U``. Defaults to
            ``adjacency`` (anchors share the grid's neighborhood structure).
        n_outer: Maximum relational refinement iterations.
        n_inner: Sinkhorn iterations per refinement.
        tol: Convergence tolerance on successive couplings / Sinkhorn updates.

    Returns:
        The ``(n, n)`` soft coupling matrix ``M*``.
    """
    S = np.asarray(similarity, dtype=np.float64)
    n, m = S.shape

    a_v = row_normalize(adjacency)
    a_u = row_normalize(adjacency if adjacency_anchor is None else adjacency_anchor)

    log_a = np.full(n, -np.log(n))
    log_b = np.full(m, -np.log(m))

    beta = float(beta)
    lam = float(lam)

    # Initialize with the feature-only entropic OT coupling. Potentials are
    # warm-started across refinements so each (sharp) Sinkhorn solve converges
    # quickly from the previous solution.
    coupling, f, g = sinkhorn_plan(
        beta * S, log_a, log_b, n_iter=n_inner, tol=tol, return_potentials=True
    )

    log_kernel = beta * S
    for _ in range(n_outer):
        # Row-stochastic view of the current assignment for stable propagation.
        rowsum = coupling.sum(axis=1, keepdims=True)
        p = np.divide(
            coupling, rowsum, out=np.zeros_like(coupling), where=rowsum > 1e-300
        )
        relational = a_v @ p @ a_u.T  # (A_V M A_U^T), entries in [0, 1]

        log_kernel = beta * (S + lam * relational)
        new_coupling, f, g = sinkhorn_plan(
            log_kernel, log_a, log_b, n_iter=n_inner, tol=tol,
            f0=f, g0=g, return_potentials=True,
        )

        delta = np.max(np.abs(new_coupling - coupling))
        coupling = new_coupling
        if delta < tol:
            break

    # Final polish on the converged kernel so the returned coupling satisfies
    # the marginal constraints (doubly stochastic) to tolerance. Warm-started
    # from the loop, so a few hundred iterations suffice.
    coupling = sinkhorn_plan(
        log_kernel, log_a, log_b, n_iter=max(n_inner, 300), tol=1e-9, f0=f, g0=g
    )
    return coupling
