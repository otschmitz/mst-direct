"""k-nearest-neighbor spatial adjacency construction (Algorithm 1, step 7).

A symmetric k-NN graph over the spatial coordinates captures the local spatial
structure: each point is connected to its ``k`` closest neighbors by Euclidean
distance. The relational matching term (see :mod:`mst_direct.relational`) uses
this graph to encourage spatially adjacent points to map to spatially adjacent
anchors, which is what preserves the variogram structure.
"""

from __future__ import annotations

import numpy as np

__all__ = ["knn_adjacency", "pairwise_distances", "row_normalize"]


def pairwise_distances(coords: np.ndarray) -> np.ndarray:
    """Euclidean distance matrix of shape ``(n, n)`` from ``(n, d)`` coords."""
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2:
        raise ValueError("coords must be a 2-D array of shape (n, d)")
    # ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 x_i . x_j
    sq = np.sum(coords**2, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * coords @ coords.T
    np.maximum(d2, 0.0, out=d2)
    return np.sqrt(d2)


def knn_adjacency(
    coords: np.ndarray,
    k: int = 8,
    symmetric: bool = True,
    dtype=np.float64,
) -> np.ndarray:
    """Build a binary k-NN adjacency matrix ``A`` (Algorithm 1, step 7).

    Args:
        coords: ``(n, d)`` array of spatial coordinates.
        k: Number of nearest neighbors per point (the paper uses ``k = 8``).
        symmetric: If true, symmetrize via ``A = max(A, A.T)`` so the edge set
            is undirected (the union of both points' neighbor lists).
        dtype: Output dtype.

    Returns:
        ``(n, n)`` adjacency matrix with zero diagonal. ``A[i, j] = 1`` iff
        ``j`` is among the ``k`` nearest neighbors of ``i`` (or vice versa when
        ``symmetric``).
    """
    coords = np.asarray(coords, dtype=np.float64)
    n = coords.shape[0]
    if k <= 0:
        raise ValueError("k must be a positive integer")
    k = min(k, n - 1)

    dist = pairwise_distances(coords)
    np.fill_diagonal(dist, np.inf)  # never connect a point to itself

    # Indices of the k smallest distances per row.
    nn = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]
    adj = np.zeros((n, n), dtype=dtype)
    rows = np.repeat(np.arange(n), k)
    adj[rows, nn.ravel()] = 1.0

    if symmetric:
        adj = np.maximum(adj, adj.T)
    return adj


def row_normalize(adj: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return a row-stochastic copy of ``adj`` (rows summing to 1).

    Isolated rows (all zeros) are left as zeros rather than producing NaNs.
    """
    adj = np.asarray(adj, dtype=np.float64)
    rowsum = adj.sum(axis=1, keepdims=True)
    return np.divide(adj, rowsum, out=np.zeros_like(adj), where=rowsum > eps)
