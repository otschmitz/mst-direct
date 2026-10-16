"""Entropy-regularized optimal transport via the Sinkhorn algorithm.

Implements equations (3)-(6) of the paper. The entropic OT problem

    min_M  <C, M> - (1/beta) H(M),   H(M) = -sum_ij M_ij log M_ij     (eq. 5)

subject to the marginal constraints ``sum_j M_ij = a_i`` and
``sum_i M_ij = b_j`` (uniform ``1/n`` by default, eq. 4) has the solution

    M = diag(r) K diag(c),   K = exp(-beta * C)                        (eq. 6)

where the scaling vectors ``r``, ``c`` are found by alternating row/column
normalization. Larger ``beta`` (smaller regularization ``1/beta``) yields
sharper couplings closer to the unregularized optimal transport plan.

All iterations run in the log domain for numerical stability, so large
``beta`` values (the paper uses ``beta = 35``) do not overflow.
"""

from __future__ import annotations

import numpy as np

__all__ = ["logsumexp", "sinkhorn", "sinkhorn_plan"]


def logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    """Numerically stable ``log(sum(exp(a)))`` along ``axis`` (keepdims)."""
    a = np.asarray(a, dtype=np.float64)
    a_max = np.max(a, axis=axis, keepdims=True)
    a_max = np.where(np.isfinite(a_max), a_max, 0.0)
    out = np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True)) + a_max
    return out


def sinkhorn_plan(
    log_kernel: np.ndarray,
    log_a: np.ndarray | None = None,
    log_b: np.ndarray | None = None,
    n_iter: int = 200,
    tol: float = 1e-9,
    f0: np.ndarray | None = None,
    g0: np.ndarray | None = None,
    return_potentials: bool = False,
):
    """Project an (unnormalized) log-kernel onto the transport polytope.

    Given ``log K`` (the *exponent* matrix, i.e. ``-beta C`` plus any extra
    log-domain rewards), find scaling potentials ``f``, ``g`` such that

        M = exp(f[:, None] + log_kernel + g[None, :])

    has row marginals ``exp(log_a)`` and column marginals ``exp(log_b)``.

    Args:
        log_kernel: ``(n, m)`` exponent matrix.
        log_a: Log of target row marginals (default uniform ``-log n``).
        log_b: Log of target column marginals (default uniform ``-log m``).
        n_iter: Maximum number of Sinkhorn iterations.
        tol: Convergence tolerance on the column potential update.
        f0: Optional warm-start for the row potential (length ``n``).
        g0: Optional warm-start for the column potential (length ``m``).
        return_potentials: If true, return ``(M, f, g)`` so the potentials can
            warm-start a subsequent solve (used by the relational refinement).

    Returns:
        The ``(n, m)`` coupling matrix ``M`` (or ``(M, f, g)``).
    """
    log_kernel = np.asarray(log_kernel, dtype=np.float64)
    n, m = log_kernel.shape
    if log_a is None:
        log_a = np.full(n, -np.log(n))
    if log_b is None:
        log_b = np.full(m, -np.log(m))
    log_a = np.asarray(log_a, dtype=np.float64).ravel()
    log_b = np.asarray(log_b, dtype=np.float64).ravel()

    f = np.zeros(n) if f0 is None else np.asarray(f0, dtype=np.float64).copy()
    g = np.zeros(m) if g0 is None else np.asarray(g0, dtype=np.float64).copy()
    for _ in range(n_iter):
        f = log_a - logsumexp(log_kernel + g[None, :], axis=1).ravel()
        g_new = log_b - logsumexp(log_kernel + f[:, None], axis=0).ravel()
        if np.max(np.abs(g_new - g)) < tol:
            g = g_new
            break
        g = g_new

    coupling = np.exp(f[:, None] + log_kernel + g[None, :])
    if return_potentials:
        return coupling, f, g
    return coupling


def sinkhorn(
    cost: np.ndarray,
    beta: float,
    a: np.ndarray | None = None,
    b: np.ndarray | None = None,
    n_iter: int = 200,
    tol: float = 1e-9,
) -> np.ndarray:
    """Solve the entropic OT problem (eq. 5-6) for a given cost matrix.

    Args:
        cost: ``(n, m)`` cost matrix ``C``.
        beta: Inverse regularization strength (kernel ``K = exp(-beta C)``).
        a: Source marginals (default uniform ``1/n``).
        b: Target marginals (default uniform ``1/m``).
        n_iter: Maximum Sinkhorn iterations.
        tol: Convergence tolerance.

    Returns:
        The optimal coupling ``M`` of shape ``(n, m)``.
    """
    cost = np.asarray(cost, dtype=np.float64)
    log_a = None if a is None else np.log(np.asarray(a, dtype=np.float64))
    log_b = None if b is None else np.log(np.asarray(b, dtype=np.float64))
    return sinkhorn_plan(-float(beta) * cost, log_a, log_b, n_iter=n_iter, tol=tol)
