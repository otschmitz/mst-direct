"""Greedy rounding of a soft coupling to a hard permutation (eq. 9-10).

The relational Sinkhorn coupling ``M*`` is a soft (doubly-stochastic) matrix.
To obtain a valid simulation we need a *permutation* ``pi`` assigning each
source point to a unique anchor. We round greedily, processing points in order
of decreasing confidence

    confidence_i = max_j M*_ij                                          (eq. 9)

and assigning each to its highest-probability anchor that is still free

    pi(i) = argmax_{j not used} M*_ij                                   (eq. 10)

Prioritizing high-confidence matches reduces conflict propagation to
lower-confidence assignments. The result is a complete bijection, which is
exactly what makes MST-Direct preserve the joint distribution: the output is a
spatial *rearrangement* of the original value tuples.
"""

from __future__ import annotations

import numpy as np

__all__ = ["greedy_round"]


def greedy_round(coupling: np.ndarray) -> np.ndarray:
    """Round a soft coupling matrix to a hard permutation (eq. 9-10).

    Args:
        coupling: ``(n, n)`` soft assignment matrix ``M*``.

    Returns:
        Integer array ``pi`` of length ``n`` where ``pi[i]`` is the anchor
        index assigned to source point ``i``. Every anchor is used exactly
        once (``pi`` is a permutation of ``range(n)``).
    """
    coupling = np.asarray(coupling, dtype=np.float64)
    n, m = coupling.shape
    if n != m:
        raise ValueError(f"coupling must be square to yield a permutation, got {coupling.shape}")

    confidence = coupling.max(axis=1)
    order = np.argsort(-confidence, kind="stable")  # decreasing confidence

    perm = np.full(n, -1, dtype=np.int64)
    used = np.zeros(n, dtype=bool)
    for i in order:
        row = coupling[i].copy()
        row[used] = -np.inf
        j = int(np.argmax(row))
        perm[i] = j
        used[j] = True
    return perm
