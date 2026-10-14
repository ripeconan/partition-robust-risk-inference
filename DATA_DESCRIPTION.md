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
