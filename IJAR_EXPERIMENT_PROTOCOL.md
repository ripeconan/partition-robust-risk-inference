# IJAR revision: uncertainty propagation experiments

Locked 2026-10-14 before the new numerical results were inspected.

Purpose: assess the general q-of-r result on real retained scores and quantify
how recovering item identifiers reduces grouping uncertainty. No new detector
training, no new labels, and no post-hoc choice of the best vote.

Data: the existing SHA-256-verified 815-run score bundle. Select all 120 AD2
Test-S runs with independent normal-calibration 95th-percentile thresholds, three
scorers, eight categories and seeds 0--4. This matches the original 120-cell
partition experiment. Use normal items and the same locked thresholds.

Experiment A: evaluate every q=1,...,5 separately. Compare the verified count
with the sharp count-plus-cap identification set. Report set width, one-sided
bound envelope, and C/V/U at alpha=delta=0.05. The q=1 results must reproduce
the archived 64 robust and 84 mapped violations. These are descriptive,
correlated sensitivity cells, not independent replications or a new family of
simultaneously certified claims. Increasing q also changes missed-defect risk;
this analysis does not select q or authorize release.

Experiment B: recover complete item groups at fractions 0, .25, .5, .75, 1
(floor(fraction*N)), using 20 outcome-independent hash orderings. Each ordering
is nested. Add the known rejected-item count to the sharp residual count set.
Maintain the original total N in the binomial calculation; recovered IDs add
information but no new verification items. Report mean set width, mean bound
width and directional decision agreement with full mapping. All IDs are
genuine filename-derived IDs; masking is synthetic metadata loss on real
scores. Require nested sets and no directional contradiction at each step.

Experiment C: independently enumerate attainable triples (N,A,K) by dynamic
programming over all item sizes and alarm counts, for F=1,...,16, r=1,...,6,
q=1,...,r, and compare with the claimed closed form. This is a finite
computational check of sharpness, separate from the general proof.

Resources: CPU only, <=2 BLAS/OpenMP threads, nice +10 on ssh inner. Expected
minutes, <1 GB RAM and <100 MB output. Stop on a checksum error, incompatible
group sizes, original-result mismatch, geometric discrepancy or nesting
failure. Preserve all outputs and the exact script hash, environment, timing
and exit status. No external publication or repository push is authorized by
this protocol.

Pre-run correction: the first execution stopped before computing results
because the initially specified raw 3-sigma arm contains only 24 runs. Manifest
inspection confirmed that the archived 120-cell analysis uses the normal-image
quantile arm. The selector was corrected to that arm; the 64/84 reproduction
check remains unchanged. The failed log is retained.

Input audit correction: Test-S has up to five views per item, rather than
exactly five in every category. The first group-size assertion caught
fruit_jelly with three views per item. The residual analysis uses the actual
number of remaining frames and the conservative cap r=5, exactly as required
by the theorem. This correction was made before any results were summarized.
