# Reproduction data and provenance

This release contains the processed inputs and outputs needed to reproduce the paper's numerical claims. All experiments use existing public industrial-inspection benchmarks. It does not include benchmark images or trained checkpoints, and it does not grant rights to those third-party materials. Obtain original images from the providers under their own terms.

## Included files

| Path | Contents and unit |
|---|---|
| `artifacts/score_bundle/run_scores.npz` | Compressed arrays of calibration and evaluation scores, labels and sanitized relative sample identifiers for the released runs |
| `artifacts/score_bundle/run_manifest.jsonl` | Run-level provenance and configuration records for 815 score runs |
| `artifacts/score_bundle/score_bundle_manifest.json` | Input sizes, SHA-256 checksums and 2,414 derived metric-cell count |
| `artifacts/fidelity_points.csv` | Frozen table-level export, 2,118 rows; its historical scope is described in `analysis_manifest.json` and the README |
| `artifacts/ad2_partition_observations.csv` | Grouping-level counts and risk verdicts |
| `artifacts/reliability_extension/` | Dual-risk, metadata, batch-dependence and voting summaries |
| `artifacts/ijar_uncertainty/vote_cells.csv` | 600 configuration-vote cells: 120 AD2 Test-S configurations and five voting thresholds |
| `artifacts/ijar_uncertainty/recovery_cells.csv` | 60,000 recovery evaluations: the same 600 cells, 20 nested recovery orders and five fractions |
| `artifacts/ijar_uncertainty/vote_summary.csv` | Summary by voting threshold |
| `artifacts/ijar_uncertainty/recovery_summary.csv` | Summary by voting threshold and target recovery fraction |
| `artifacts/ijar_uncertainty/summary.json` | Analysis results, environment, input/code hashes, and independent finite geometry check (26,523 configurations) |

CSV tables use a header row. The analytical code specifies how each column is computed. See `IJAR_EXPERIMENT_PROTOCOL.md` for the calibration arm, item identifiers, recovery masks, voting rules and resource settings. The physical item is the verification unit. Repeated detector seeds and recovery masks reuse observations; their rows must not be treated as new independent verification samples.

## Reuse and integrity

The code and accompanying authored documentation retain the package's MIT license. The released numerical tables and processed scores are provided as part of the reproduction package; upstream benchmark terms continue to apply to upstream materials. Raw benchmark imagery is not redistributed.

`MANIFEST.sha256` covers the packaged files except itself. The release-level `SHA256SUMS.txt` covers the downloadable ZIP. Verify these before reproduction. The frozen outputs record the original run environment; a fresh run can change environment fields, elapsed time and PDF metadata while reproducing the same numerical results.

## Validation record

The existing numerical reproduction checked 815 score runs and 2,414 derived metric cells, the partition and dependence simulations, and all 21 unit tests. The new recovery analysis ran on the designated experiment server and found no contradictions with complete grouping; the independent finite geometry check found no mismatches. Retained and new figures were generated and checked separately.

The initial all-in-one validation encountered a NumPy/Matplotlib binary conflict in a shared system environment at the plotting stage. Numerical stages and tests passed; plotting passed separately after isolating user-site packages. Use a fresh environment with compatible dependencies. This release does not describe that initial run as a successful uninterrupted full pipeline.

## v1.1.0 extension, 2026-10-15

The same frozen score bundle supplies the new comparisons. No new detector fitting, benchmark images, independent verification items or private industrial observations are added. `artifacts/strengthening/summary.json` records the executed analysis source hash and environment. The formal output archive is also retained locally with SHA-256 1aa7d7b1f38d8f328e9ece2646b6e0e849118e87a14231a2ebb6add36271452d.

`threshold_optimality_checks.csv` tests size-calibrated monotone thresholds, explicitly retaining exact-binomial-tail ties. `real_comparisons.csv` reports all method/budget/configuration/vote cells; `real_summary.csv` summarizes them. `solver_timings.csv` times count endpoints only, excluding detector scoring and CP evaluation. `exact_power_grid.csv` uses exact convolution, and `recovery_frontier.csv` uses exact binomial summation; neither is a Monte Carlo estimate. Small binary matrices independently validate the integer view-margin endpoints. The real full-mapping reference is available only for evaluation; alternative methods receive only their declared aggregate or view metadata.

The recovery-frontier theorem assumes random recovery independent of the entire item alarm vector. Its critical fractions are asymptotic planning quantities; finite-sample certification still uses the reported CP rule. The monotone optimality claim applies to complete fixed-view designs and deterministic threshold rules. The strict lower-confidence-bound convention is conservative at exact tail ties. All mechanisms, sample sizes and risk budgets in the protocol are reported, including unresolved regimes.

## v1.2.0 extension

The complete-family analysis reuses the exact same score-derived integer endpoints. It adds no benchmark, image, label or independent physical item. Family memberships are fixed before this recalculation; the earlier marginal results had already been observed, so the protocol labels the analysis retrospective. At a fixed risk threshold, false certification and false violation correspond to different truth states. A union bound over targets, using CP level delta/M, controls any wrong directional statement; truthful metadata refinements share the mapped error events and do not add another method-count factor. This does not license unrestricted data-dependent target construction or arbitrary additional risk thresholds.
