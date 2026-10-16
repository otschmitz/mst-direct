"""Scalable MST-Direct matching for large grids.

``mst_direct.simulator.MSTDirect`` is dense ``O(n^2)`` in memory and ``~O(n^3)``
per relational iteration, so it cannot run at the DMS grid sizes (40k / 10k
points). This module keeps MST-Direct's idea — *Matching via Sinkhorn
Transport*: a permutation of the target value tuples driven by entropic optimal
transport — but makes it scale:

* the simulated field is a **bijection** onto the reference cloud, so the joint
  distribution is preserved **exactly** (the MST-Direct guarantee);
* the spatial structure is carried by an FFT-MA Gaussian **backbone** with the
  prescribed variogram (Section 4 of DMS draws independent Gaussian fields the
  same way), so the matching only has to map the cloud onto that template;
* the transport is solved by a **sparse, candidate-restricted Sinkhorn** in the
  log domain. Each grid location only competes for its ``n_candidates`` nearest
  reference tuples (KD-tree), giving ``O(n * n_candidates)`` memory instead of
  ``O(n^2)``;
* a greedy rounding turns the soft coupling into a permutation, and any leftover
  locations are repaired by nearest-unused assignment so the result is always a
  complete bijection;
* **hard-data conditioning** is supported by *pinning* chosen locations to
  chosen tuples before matching (used by the conditional experiment).

Cost is squared Euclidean distance in the per-variable standardized space
(quantile-style matching of extremes to extremes), with Gibbs kernel
``K = exp(-beta * cost)``.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import norm, rankdata

__all__ = ["ScalableMST", "mst_simulate", "transport_features"]


def _standardize(a: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = a.mean(axis=0)
    sd = a.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    return (a - mu) / sd, mu, sd


def transport_features(a: np.ndarray, transform: str = "whiten") -> np.ndarray:
    """Return features used only for transport costs.

    The original tuples are never transformed on output.  Whitening is the
    appropriate default for an independent Gaussian backbone: unlike marginal
    standardization, it removes the strong linear dependence that otherwise
    makes much of the sparse candidate graph unreachable.
    """
    z, _, _ = _standardize(a)
    if transform == "standardize":
        return z
    if transform == "gaussian_copula_whiten":
        n = len(z)
        z = np.column_stack(
            [norm.ppf((rankdata(z[:, c], method="average") - 0.5) / n)
             for c in range(z.shape[1])]
        )
    elif transform != "whiten":
        raise ValueError(
            "feature_transform must be 'standardize', 'whiten', or "
            "'gaussian_copula_whiten'"
        )
    if z.shape[1] == 1:
        return z
    z = z - z.mean(axis=0)
    evals, evecs = np.linalg.eigh(np.cov(z, rowvar=False))
    whitener = evecs @ np.diag(np.clip(evals, 1e-10, None) ** -0.5) @ evecs.T
    out = z @ whitener
    return (out - out.mean(axis=0)) / np.where(out.std(axis=0) > 0, out.std(axis=0), 1.0)


def _segment_logsumexp(values: np.ndarray, seg: np.ndarray, n_seg: int) -> np.ndarray:
    """log-sum-exp of ``values`` grouped by integer segment id ``seg``.

    Returns a length-``n_seg`` array; empty segments get ``-inf``.
    """
    seg_max = np.full(n_seg, -np.inf)
    np.maximum.at(seg_max, seg, values)
    safe = np.where(np.isfinite(seg_max), seg_max, 0.0)
    contrib = np.exp(values - safe[seg])
    seg_sum = np.zeros(n_seg)
    np.add.at(seg_sum, seg, contrib)
    out = np.full(n_seg, -np.inf)
    nz = seg_sum > 0
    out[nz] = np.log(seg_sum[nz]) + safe[nz]
    return out


def _sparse_sinkhorn(
    rows: np.ndarray,
    cols: np.ndarray,
    log_kernel: np.ndarray,
    n: int,
    m: int,
    n_iter: int,
    tol: float,
) -> np.ndarray:
    """Log-domain Sinkhorn on a sparse edge list. Returns coupling per edge.

    Marginals are uniform (``1/n`` rows, ``1/m`` cols). Edges not present are
    treated as zero coupling.
    """
    log_a = -np.log(n)
    log_b = -np.log(m)
    f = np.zeros(n)
    g = np.zeros(m)
    for _ in range(n_iter):
        # row update: f_i = log_a - LSE_j (logK_ij + g_j)
        f = log_a - _segment_logsumexp(log_kernel + g[cols], rows, n)
        f[~np.isfinite(f)] = 0.0
        g_new = log_b - _segment_logsumexp(log_kernel + f[rows], cols, m)
        g_new[~np.isfinite(g_new)] = 0.0
        delta = np.max(np.abs(g_new - g))
        g = g_new
        if delta < tol:
            break
    return np.exp(f[rows] + log_kernel + g[cols])


def _greedy_round(
    rows: np.ndarray,
    cols: np.ndarray,
    weight: np.ndarray,
    n: int,
    m: int,
) -> np.ndarray:
    """Greedy bijection from a sparse soft coupling.

    Processes rows in order of descending best-edge weight; each takes its
    highest-weight column still free. Returns ``assignment`` (length ``n``,
    ``-1`` where no free candidate remained).
    """
    order_in_row = np.lexsort((-weight, rows))  # group by row, best weight first
    # best weight per row for the processing order
    row_best = np.full(n, -np.inf)
    np.maximum.at(row_best, rows, weight)
    row_order = np.argsort(-row_best, kind="stable")

    # candidate columns/weights grouped by row
    start = np.searchsorted(rows[order_in_row], np.arange(n), side="left")
    end = np.searchsorted(rows[order_in_row], np.arange(n), side="right")
    sorted_cols = cols[order_in_row]

    assignment = np.full(n, -1, dtype=np.int64)
    used = np.zeros(m, dtype=bool)
    for i in row_order:
        for e in range(start[i], end[i]):
            j = sorted_cols[e]
            if not used[j]:
                assignment[i] = j
                used[j] = True
                break
    return assignment


def _repair_leftovers(
    assignment: np.ndarray,
    B: np.ndarray,
    U: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Assign leftovers by repeated nearest-free matching.

    Every repaired assignment is selected by feature distance.  This replaces
    the old arbitrary "first remaining column" fallback.
    """
    free_rows = np.where(assignment < 0)[0]
    if len(free_rows) == 0:
        return assignment, {"repaired": 0, "repair_rounds": 0, "repair_max_distance": 0.0}
    initial = len(free_rows)
    used = np.zeros(len(U), dtype=bool)
    used[assignment[assignment >= 0]] = True
    free_cols = np.where(~used)[0]
    rounds = 0
    max_distance = 0.0
    while len(free_rows):
        rounds += 1
        tree = cKDTree(U[free_cols])
        kq = min(len(free_cols), 32)
        dist, nn = tree.query(B[free_rows], k=kq)
        dist = np.asarray(dist).reshape(len(free_rows), kq)
        nn = np.asarray(nn).reshape(len(free_rows), kq)
        # Most constrained/closest rows choose first.  Collisions retry another
        # candidate in this round; unresolved rows are reconsidered against a
        # rebuilt tree, so no row is ever assigned to an unrelated first-free
        # column.
        row_order = np.argsort(dist[:, 0], kind="stable")
        taken = np.zeros(len(free_cols), dtype=bool)
        placed_rows = np.zeros(len(free_rows), dtype=bool)
        for q in row_order:
            for rank, local in enumerate(nn[q]):
                if not taken[local]:
                    assignment[free_rows[q]] = free_cols[local]
                    taken[local] = True
                    placed_rows[q] = True
                    max_distance = max(max_distance, float(dist[q, rank]))
                    break
        free_rows = free_rows[~placed_rows]
        free_cols = free_cols[~taken]
    return assignment, {
        "repaired": int(initial),
        "repair_rounds": int(rounds),
        "repair_max_distance": float(max_distance),
    }


class ScalableMST:
    """Scalable Matching via Sinkhorn Transport.

    Args:
        beta: Inverse Sinkhorn regularization (kernel sharpness). Larger ->
            sharper / closer to true OT. Default 6.0 in standardized units.
        n_candidates: Candidate reference tuples considered per location.
        max_candidates: Optional adaptive upper bound. Candidate support doubles
            until at most ``candidate_zero_tolerance`` of target columns have
            zero degree, or this bound is reached.
        candidate_zero_tolerance: Maximum desired zero-degree target fraction.
        n_iter: Maximum Sinkhorn iterations.
        tol: Convergence tolerance on the column potentials.
        random_state: Seed (reserved for tie-breaking / future stochastic use).
        feature_transform: Cost-space transform. ``"whiten"`` (default) aligns
            dependent tuple features with independent Gaussian backbones while
            output values remain untouched.
    """

    def __init__(
        self,
        beta: float = 6.0,
        n_candidates: int = 48,
        max_candidates: int | None = None,
        candidate_zero_tolerance: float = 0.01,
        n_iter: int = 150,
        tol: float = 1e-6,
        k_relational: int = 8,
        lam_relational: float = 0.0,
        n_relax: int = 0,
        feature_transform: str = "whiten",
        random_state: int | None = None,
    ) -> None:
        self.beta = float(beta)
        self.n_candidates = int(n_candidates)
        self.max_candidates = int(max_candidates) if max_candidates is not None else int(n_candidates)
        self.candidate_zero_tolerance = float(candidate_zero_tolerance)
        self.n_iter = int(n_iter)
        self.tol = float(tol)
        self.k_relational = int(k_relational)
        self.lam_relational = float(lam_relational)
        self.n_relax = int(n_relax)
        self.feature_transform = str(feature_transform)
        self.random_state = random_state
        self.assignment_: np.ndarray | None = None
        self.diagnostics_: dict[str, float | int | str] = {}

    def _match(
        self, query: np.ndarray, Uf: np.ndarray, tree: cKDTree, nf: int
    ) -> np.ndarray:
        """One candidate-Sinkhorn-round pass mapping ``query`` rows to ``Uf``."""
        C = min(self.n_candidates, nf)
        max_c = min(max(self.max_candidates, C), nf)
        while True:
            dist, nn = tree.query(query, k=C)
            nn_check = np.asarray(nn).reshape(nf, C)
            zero_fraction = 1.0 - np.unique(nn_check).size / nf
            if zero_fraction <= self.candidate_zero_tolerance or C >= max_c:
                break
            C = min(2 * C, max_c)
        dist = np.atleast_2d(dist)
        nn = np.atleast_2d(nn)
        rows = np.repeat(np.arange(nf), C)
        cols = nn.ravel()
        cost = (dist.ravel()) ** 2
        cost = cost / (np.median(cost) + 1e-12)  # beta is grid-size independent
        log_kernel = -self.beta * cost
        coupling = _sparse_sinkhorn(rows, cols, log_kernel, nf, nf, self.n_iter, self.tol)
        local = _greedy_round(rows, cols, coupling, nf, nf)
        local, repair = _repair_leftovers(local, query, Uf)
        selected_distance = np.linalg.norm(query - Uf[local], axis=1)
        self.diagnostics_ = {
            "feature_transform": self.feature_transform,
            "candidate_count": int(C),
            "candidate_count_requested": int(self.n_candidates),
            "candidate_columns_zero_degree": int(nf - np.unique(cols).size),
            "greedy_unmatched": int(repair["repaired"]),
            "greedy_unmatched_fraction": float(repair["repaired"] / max(nf, 1)),
            "repair_rounds": int(repair["repair_rounds"]),
            "repair_max_distance": float(repair["repair_max_distance"]),
            "final_match_distance_mean": float(selected_distance.mean()),
            "final_match_distance_max": float(selected_distance.max()),
        }
        return local

    def simulate(
        self,
        backbone: np.ndarray,
        cloud: np.ndarray,
        pinned: tuple[np.ndarray, np.ndarray] | None = None,
        coords: np.ndarray | None = None,
    ) -> np.ndarray:
        """Map the reference ``cloud`` onto the spatial ``backbone``.

        Args:
            backbone: ``(n, d)`` spatially-structured template (e.g. FFT-MA
                Gaussian fields with the prescribed variogram).
            cloud: ``(n, d)`` reference target tuples (the joint distribution).
                Must have the same number of rows as ``backbone`` so the result
                is a bijection (resample the reference cloud to ``n`` first).
            pinned: Optional ``(rows, cloud_rows)`` hard assignments (equal
                length). Those grid rows output exactly ``cloud[cloud_rows]``
                and are excluded from the transport. Used for conditioning.
            coords: ``(n, 2)`` grid coordinates. Required when
                ``lam_relational > 0`` and ``n_relax > 0`` to build the spatial
                k-NN graph for the relational refinement (MST-Direct's
                relational term), which reduces short-scale roughness.

        Returns:
            ``(n, d)`` simulated field — a permutation of ``cloud`` (joint
            distribution preserved exactly).
        """
        B = np.asarray(backbone, dtype=np.float64)
        cloud = np.asarray(cloud, dtype=np.float64)
        n, d = B.shape
        if cloud.shape[0] != n:
            raise ValueError(
                f"backbone has {n} rows but cloud has {cloud.shape[0]}; "
                "resample the cloud to match the grid size."
            )

        # Standardize both into a common space (use the cloud's statistics so
        # the matched magnitudes are comparable to the target marginals).
        Us = transport_features(cloud, self.feature_transform)
        Bs = (B - B.mean(axis=0)) / B.std(axis=0)  # backbone -> N(0,1) per dim

        assignment = np.full(n, -1, dtype=np.int64)
        free_row_mask = np.ones(n, dtype=bool)
        free_col_mask = np.ones(n, dtype=bool)
        if pinned is not None:
            prows, pcols = np.asarray(pinned[0]), np.asarray(pinned[1])
            assignment[prows] = pcols
            free_row_mask[prows] = False
            free_col_mask[pcols] = False

        free_rows = np.where(free_row_mask)[0]
        free_cols = np.where(free_col_mask)[0]
        nf = len(free_rows)
        Bf = Bs[free_rows]
        Uf = Us[free_cols]
        tree = cKDTree(Uf)

        # Initial backbone-driven match.
        local_assign = self._match(Bf, Uf, tree, nf)

        # Relational refinement (MST-Direct's k-NN term, scalable form): pull
        # each location's query toward the mean assigned feature of its spatial
        # neighbors, so neighbors map to similar tuples (lower nugget).
        if self.lam_relational > 0 and self.n_relax > 0 and coords is not None:
            coords = np.asarray(coords, dtype=np.float64)
            cf = coords[free_rows]
            kg = min(self.k_relational + 1, nf)
            _, nbr = cKDTree(cf).query(cf, k=kg)
            nbr = np.atleast_2d(nbr)[:, 1:]  # drop self
            lam = self.lam_relational
            for _ in range(self.n_relax):
                neigh_feat = Uf[local_assign][nbr].mean(axis=1)  # (nf, d)
                query = (1.0 - lam) * Bf + lam * neigh_feat
                local_assign = self._match(query, Uf, tree, nf)

        assignment[free_rows] = free_cols[local_assign]
        self.assignment_ = assignment
        return cloud[assignment]


def mst_simulate(
    backbone: np.ndarray,
    cloud: np.ndarray,
    beta: float = 6.0,
    n_candidates: int = 48,
    pinned: tuple[np.ndarray, np.ndarray] | None = None,
    random_state: int | None = None,
) -> np.ndarray:
    """Functional wrapper around :class:`ScalableMST`."""
    return ScalableMST(
        beta=beta, n_candidates=n_candidates, random_state=random_state
    ).simulate(backbone, cloud, pinned=pinned)
