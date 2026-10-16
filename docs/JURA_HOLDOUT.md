# Swiss Jura conditional hold-out experiment

This real-data experiment uses the official `gstat` split of 259 training and
100 validation topsoil samples. Seven log-transformed trace-metal
concentrations (Cd, Co, Cr, Cu, Ni, Pb, Zn) are simulated jointly. Validation
concentrations are not accessed until all 30 realizations have been generated.

For each MST-Direct realization, the target cloud contains all 259 conditioning
tuples plus 100 tuples sampled with replacement from training only. The former
are pinned at their observed locations; the latter are assigned to validation
locations. This preserves the generated target cloud, not the unknown field
population. PPMT and an independent normal-score conditional simulator use the
same training split and spherical covariance protocol.

## Main results

| Method | Mean 90% coverage | Mean log-CRPS | Standardized energy score |
|---|---:|---:|---:|
| MST-Direct | 0.844 | 0.2544 | 1.5709 |
| PPMT | 0.851 | 0.2543 | 1.5681 |
| Independent normal score | 0.757 | 0.2649 | 1.6553 |

The paired site-bootstrap difference in energy score was 0.0028 (95% interval
-0.0650 to 0.0736) for MST-Direct minus PPMT: the experiment does not establish
a difference between those methods. MST-Direct minus independent normal score
was -0.0844 (-0.1442 to -0.0244), favouring MST-Direct under this protocol.

MST-Direct honoured all 259 conditioning tuples exactly in all realizations.
Greedy residual repair was zero in the median realization and at most 2%; one
realization expanded support from 48 to 96 candidates. Generating the
realizations required approximately 22 seconds on the documented audit machine;
metric calculation added several seconds.

## Reproduction

```bash
python scripts/download_jura_data.py
pip install 'mst-direct[real,plot]'
python scripts/jura_holdout_experiment.py
python scripts/plot_jura_holdout.py
```

Machine-readable results are in `jura_holdout_results.json`; the summary figure
is `jura_holdout_summary.png`.

## Limitations

- Thirty realizations give coarse tail and coverage estimates.
- Spherical isotropic covariances were selected without consulting validation,
  but anisotropy and geological boundaries were not modelled.
- The target-cloud bootstrap duplicates training tuples and omits population-
  model uncertainty.
- PPMT is the repository's compact implementation, not an external reference
  package.
- Rock-stratified scores are descriptive; Portlandian has only three validation
  observations.
