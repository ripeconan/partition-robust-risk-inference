# IJAR reproduction package

This package accompanies **Partition-Robust Risk Inference with Incomplete Multi-View Grouping**, by Ran Wang (Hangzhou Dianzi University). It contains the retained baseline score bundle and analyses, plus the voting, identifier-recovery and independent finite-enumeration extension. The earlier anonymous snapshot does not contain these new analyses.

## Versioned data and code

Repository: https://github.com/ripeconan/partition-robust-risk-inference

Version **v1.1.0**, released 2026-10-15: https://github.com/ripeconan/partition-robust-risk-inference/releases/tag/v1.1.0

Use the attached `IJAR_Reproduction_Code.zip` and its SHA-256 checksum for the exact paper snapshot. `CITATION.cff` and `CITATION.bib` provide citation metadata. `DATA_DESCRIPTION.md` describes the included processed data and their provenance. This is a research artifact release, not a claim of journal acceptance.

## Version 1.1 strengthening analyses

Run `python strengthening_analysis.py` for the aggregate/MILP/LP, retained-view-margin and Hoeffding comparisons, exact decision-power grid and identifier-recovery frontier. Run `python plot_strengthening.py` to render the two additional figures. This is a separate command from the retained baseline `reproduce_headline.py` workflow.

`STRENGTHENING_PROTOCOL.txt` records the initial protocol and explicit boundary/theory amendments. `artifacts/strengthening/` contains 600 aggregate threshold checks (15 exact-tail ties), 364 independently enumerated binary-margin cases, 600 real configuration-vote comparisons, 16,800 method-budget rows, 4,320 exact-power rows and 48 frontier rows. These rows reuse observations and do not increase the physical sample size. All reported integer endpoint comparisons agree. The original, corrected and final execution logs are retained.

The final analysis used Python 3.10.12, NumPy 2.2.6 and SciPy 1.15.3 on an i9-13900K CPU, with two threads, approximately 3.7 seconds process time and 128 MiB peak RSS. No GPU or detector retraining is needed. `requirements-analysis-lock.txt` pins the new numerical analysis dependencies; the original `requirements-lock.txt` records the earlier plotting/reproduction environment. Runtime and solver timings depend on the machine. The existing 21 unit tests also pass under the new analysis environment.

On Linux, a bounded CPU invocation is:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' python strengthening_analysis.py
```

The older v1.0.0 release remains available and immutable. Use v1.1.0 for the revised manuscript containing sharp aggregate error curves, the recovery frontier and view-margin comparisons.

## IJAR extension

Run `python ijar_uncertainty_analysis.py` to reproduce 600 configuration--vote cells, 60,000 metadata-recovery evaluations and the 26,523-case independent geometry check. See `IJAR_EXPERIMENT_PROTOCOL.md`. The actual number of shifted-light views is used, with cap five; some categories contain three views. Run `python make_ijar_figures.py` to generate the new workflow and recovery figures.

`artifacts/ijar_uncertainty/` contains the full CSV outputs and `summary.json`, including input and script hashes and the recorded computing environment. The 20 masks and all detector seeds reuse observations; do not treat them as new independent verification samples.

Use the isolated environment described below. A system environment with NumPy 2 and a Matplotlib binary compiled for NumPy 1 can fail at plotting; install the declared plotting requirements in a fresh virtual environment. The retained baseline plotting script has been corrected to resolve data within this package.

## Requirements

- Python 3.9 or later
- packages listed in requirements.txt

Create an isolated environment and install the dependencies:

~~~bash
python -m venv .venv
~~~

Activate the environment using the command appropriate for the operating
system (`source .venv/bin/activate` on Linux/macOS, or
`.venv\Scripts\Activate.ps1` in Windows PowerShell), then run
`python -m pip install -r requirements.txt`.

## One-command headline reproduction

From the repository root, run:

~~~bash
python reproduce_headline.py
~~~

The command verifies the frozen table and score bundles, reproduces the
three-seed and partition analyses, reruns both 100,000-replicate simulations,
regenerates the retained baseline and new IJAR figures in `artifacts/figures/`, and executes
the unit tests. It requires no GPU and normally completes in roughly 1--3
minutes on a recent desktop CPU, excluding environment installation. Runtime
depends on the CPU and file-system cache. For a non-paper smoke test, reduce
the simulation count explicitly, for example
`python reproduce_headline.py --replicates 10000`; only the default 100,000
replicates reproduce the reported Monte Carlo values.

The released score bundle records the software and hardware used for the
original detector runs. Those detector runs are not repeated by this compact
validation package.

## Reproduce the reported counts

Run the following command from the repository root:

~~~bash
python reproduce_tables.py
~~~

The script performs three checks before printing the table entries:

1. required table cells and dataset/model/category counts are present;
2. semantic keys are unique and the declared inference convention is unchanged;
3. every stored verdict agrees with a fresh calculation from
   (fp, n0, alpha, delta).

The expected marginal counts are:

| dataset or comparison | FPR budget | certified | violated | unresolved |
|---|---:|---:|---:|---:|
| MVTec AD 2 | 5% | 0 | 0 | 24 |
| MVTec AD | 5% | 1 | 13 | 31 |
| VisA | 5% | 1 | 3 | 32 |
| MVTec AD 2 | 1% | 0 | 2 | 22 |
| MVTec AD | 1% | 0 | 24 | 21 |
| VisA | 1% | 0 | 22 | 14 |
| AD2 PatchCore, empirical quantile | 5% | 0 | 0 | 8 |
| AD2 PatchCore, raw mean + 3 sigma | 5% | 0 | 6 | 2 |

## Rederive the validation results from released scores

~~~bash
python reproduce_from_scores.py
python summarize_multiseed.py
~~~

This command starts from the compact score bundle in
`artifacts/score_bundle/`, independently recomputes the locked thresholds,
frame-to-item aggregation, false-positive counts, exact bounds, and headline
three-state totals. The second command verifies the expected primary cells at
seeds 0--2 and reports seed-wise verdict counts and stability patterns. The
bundle manifest records 815 score runs and 2,414 derived metric cells; both
commands fail if its file sizes, SHA-256 checksums, or declared counts do not
match the released files. The
bundle contains scores and sanitized relative paths, not benchmark images or
model checkpoints. If the bundle is distributed as a separate release asset,
extract it to that directory first.

## Run the planner

~~~bash
python code/sample_planner.py
~~~

The planner reports marginal and Bonferroni-adjusted verdicts, additional
zero-event items required for certification, and the minimum detectable FPR
excess. Use "python code/sample_planner.py --help" for command-line options.

## Reproduce the partition-robust analysis

~~~bash
python reproduce_partition_analysis.py
python simulate_partition_coverage.py --replicates 100000 --seed 20260726
~~~

The first command compares frame-naive, cap-only robust, count-plus-cap robust,
and verified-mapping verdicts across 120 AD2 Test-S cells. The second reproduces
the coverage stress test. The robust method ranges over unknown item counts when
necessary; when a trustworthy item count is available, it tightens the feasible
set without inferring or requiring the hidden mapping.

For a generic mapping-free validation, supply the available metadata directly:

~~~bash
python code/partition_robust.py --frames 55 --alarms 4 \
  --max-frames-per-item 5 --n-items 11 --alpha 0.05 --delta 0.05
~~~

Omit `--n-items` when the physical-item count is unknown. The command then
ranges over every item count compatible with the frame count and cap.

Use `--min-alarms-per-reject q` for a q-of-r vote. The default is q=1 and
recovers any-frame rejection. The implementation returns the sharp feasible
integer set proved in the manuscript.

## Reproduce the reliability extensions

~~~bash
python reliability_extension_analysis.py --replicates 100000 --seed 20260802
~~~

This command writes five tables to `artifacts/reliability_extension/`: the 20-seed
MVTec transistor dual-risk gate, metadata-constraint sensitivity, shared-batch
heterogeneity, q-of-r vote geometry, and a machine-readable JSON summary. The
release gate can also be called directly from `code/reliability_gate.py`.

## Run the tests

~~~bash
python -m unittest discover -s tests
~~~

## Files

~~~text
README.md
LICENSE
requirements.txt
requirements-lock.txt
reproduce_tables.py
reproduce_headline.py
reproduce_from_scores.py
summarize_multiseed.py
reproduce_partition_analysis.py
simulate_partition_coverage.py
reliability_extension_analysis.py
make_figures.py
code/
  cp_state.py
  partition_robust.py
  reliability_gate.py
  sample_planner.py
tests/
  test_core.py
artifacts/
  fidelity_points.csv
  ad2_partition_observations.csv
  analysis_manifest.json
  reliability_extension/
    dual_gate_transistor.csv
    dual_gate_summary.csv
    metadata_sensitivity.csv
    heterogeneity_sensitivity.csv
    q_of_r_geometry.csv
    summary.json
  score_bundle/
    run_manifest.jsonl
    run_scores.npz
    score_bundle_manifest.json
  figures/
    fig0_validation_architecture.pdf/png
    ...
    fig8_reliability_extensions.pdf/png
~~~

The file fidelity_points.csv contains one row per evaluated cell and claim
budget. The columns fp, n0, alpha, and delta determine the three-state verdict.
The remaining columns identify the dataset cell, threshold rule, split,
performance measurements, and stored interval values.

The two manifests have distinct scopes. `analysis_manifest.json` describes the
frozen table-level `fidelity_points.csv` export (683 source runs, 2,150 source
metric cells, and 2,118 included rows). The score-level
`score_bundle_manifest.json` describes the current all-seed production bundle
(815 runs and 2,414 derived metric cells) and records its file sizes and SHA-256
checksums. The partition-observation CSV contains only aggregate frame/item
counts and verdicts; it contains no image paths.

## Statistical rule

For fp false positives among n0 independent normal items, the one-sided
Clopper--Pearson bounds are

~~~text
upper = Beta^{-1}(1 - delta; fp + 1, n0 - fp)
lower = Beta^{-1}(delta; fp, n0 - fp + 1).
~~~

The result is certified when upper <= alpha, violated when lower > alpha, and
unresolved otherwise. At 95% one-sided confidence, zero false positives require
at least 59 independent items for a 5% FPR claim and 299 for a 1% claim.

## Scope

The package supports two reproduction levels: the compact per-cell table
recreates the manuscript tables quickly, while the score bundle rederives those
cells from released anomaly scores. It does not rerun model training or model
inference. Benchmark data remain available from their respective providers and
are not redistributed.

The package is distributed under the MIT License.
