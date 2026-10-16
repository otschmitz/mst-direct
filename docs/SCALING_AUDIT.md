# Independent scaling and sensitivity audit

This audit is not a reconstruction of the missing manuscript workflow. It uses
the public six-variable DMS analytical cloud, released `ScalableMST` code,
`beta=6`, 150 Sinkhorn iterations, no relational refinement, and 100,000 sampled
pairs per variable for variograms. Exact rows are in
`scaling_audit_summary.csv`; the executable protocol is `scripts/scaling_audit.py`.

## Main observations

- Every run returned a permutation, so empirical tuple preservation held.
- Median wall times across three seeds were 2.31 s at 2,500 nodes, 9.96 s at
  10,000 nodes, and 40.11 s at 40,000 nodes.
- Spatial reproduction was not stable at 40,000 nodes. The minimum
  per-variable variogram-shape correlations were 0.818, -0.012, and -0.198 for
  seeds 0, 1, and 2 respectively.
- The fraction of final assignments outside each row's original 48-candidate
  support increased with size: 9.5--11.3% at 2,500 nodes, 14.9--19.6% at 10,000
  nodes, and 29.9--36.8% at 40,000 nodes. These assignments necessarily arise
  from completion outside the sparse transport support, so the final
  permutation can differ materially from the candidate-restricted OT solution.
- At 10,000 nodes and seed 0, increasing candidates from 24 to 96 reduced the
  outside-support fraction from 29.9% to 10.8% and improved the minimum
  variogram correlation from 0.274 to 0.856, while increasing wall time from
  4.32 s to 14.90 s.
- The dimensionality check was not monotonic: at 10,000 nodes, the
  outside-support fraction was 40.4% for two variables, 22.7% for four, and
  19.6% for six in the comparable seed-0 runs. This shows that candidate-graph
  congestion depends on the geometry of the supplied cloud, not only on `d`.

## Interpretation

The experiment supports the narrow runtime and memory-motivation claims, but
contradicts a broad claim that the released default matcher reliably reproduces
the prescribed spatial correlation at 40,000 nodes. Candidate count and the
completion stage are scientifically consequential hyperparameters. The paper
must report the exact relational and candidate settings used for every figure,
the completion/outside-support rate, and multi-seed spatial metrics. A robust
implementation should expand the candidate graph adaptively until it admits a
perfect matching and should expose marginal/convergence diagnostics before
rounding.
