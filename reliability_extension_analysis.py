#!/usr/bin/env python3
"""Reliability-oriented extensions for the T-Reliability manuscript.

The script produces four auditable result tables:

1. a dual false-reject/missed-defect release gate on MVTec transistor;
2. sensitivity of partition-robust inference to metadata constraints;
3. exact-binomial sensitivity to shared-batch heterogeneity;
4. sharp feasible counts for q-of-r multi-view voting.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))

from cp_state import certify_state, cp_ci
from partition_robust import feasible_item_counts, partition_robust_verdict
from reliability_gate import dual_risk_gate


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def dual_gate_rows(source: Path) -> tuple[list[dict], list[dict]]:
    selected = []
    with source.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (
                row["dataset"] == "mvtec"
                and row["unit"] == "transistor"
                and row["threshold_rule"] == "normal_image_quantile"
                and row["split"] == "test_m"
                and math.isclose(float(row["alpha"]), 0.05)
            ):
                selected.append(row)
    rows = []
    for row in selected:
        fp = int(row["fp"])
        n0 = int(row["n0"])
        tp = int(row["tp"])
        n1 = int(row["n1"])
        result = dual_risk_gate(
            false_rejects=fp,
            normal_items=n0,
            missed_defects=n1 - tp,
            anomalous_items=n1,
            false_reject_budget=0.05,
            missed_defect_budget=0.10,
            delta=0.05,
        )
        rows.append(
            {
                "model": row["model"],
                "seed": int(row["seed"]),
                "false_rejects": fp,
                "normal_items": n0,
                "missed_defects": n1 - tp,
                "anomalous_items": n1,
                "false_reject_state": result.false_reject_state,
                "false_reject_upper": result.false_reject_upper,
                "missed_defect_state": result.missed_defect_state,
                "missed_defect_upper": result.missed_defect_upper,
                "release_verdict": result.verdict,
            }
        )
    summary = []
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["model"]].append(row)
    for model, model_rows in sorted(grouped.items()):
        states = Counter(row["release_verdict"] for row in model_rows)
        fpr_certified = sum(
            row["false_reject_state"] == "certified" for row in model_rows
        )
        summary.append(
            {
                "model": model,
                "seeds": len(model_rows),
                "fpr_certified": fpr_certified,
                "release_ready": states["release-ready"],
                "continue_testing": states["continue-testing"],
                "not_release_ready": states["not-release-ready"],
            }
        )
    return rows, summary


def envelope_for_item_counts(
    frames: int,
    alarms: int,
    maximum: int,
    item_counts: range,
    threshold: int = 1,
) -> tuple[str, float, float]:
    pairs = set()
    for items in item_counts:
        try:
            pairs.update(
                feasible_item_counts(
                    frames,
                    alarms,
                    maximum,
                    n_items=items,
                    min_alarms_per_reject=threshold,
                )
            )
        except ValueError:
            continue
    if not pairs:
        raise ValueError("metadata constraint set contains no compatible partition")
    bounds = [cp_ci(rejected, items, 0.05) for rejected, items in pairs]
    lower = min(bound[0] for bound in bounds)
    upper = max(bound[1] for bound in bounds)
    verdict = "certified" if upper <= 0.05 else "violated" if lower > 0.05 else "unresolved"
    return verdict, lower, upper


def summarize_states(
    states: np.ndarray,
    uppers: np.ndarray,
    true_rate: float,
) -> dict[str, float]:
    return {
        "upper_coverage": float(np.mean(uppers >= true_rate)),
        "certification_rate": float(np.mean(states == "certified")),
        "violation_rate": float(np.mean(states == "violated")),
        "unresolved_rate": float(np.mean(states == "unresolved")),
    }


def metadata_sensitivity(replicates: int, rng: np.random.Generator) -> list[dict]:
    items = 60
    frames_per_item = 5
    frames = items * frames_per_item
    true_rate = 0.06
    rejected_items = rng.binomial(items, true_rate, size=replicates)
    rows = []
    for mechanism, frame_alarms in (
        ("one-frame", rejected_items),
        ("all-frames", frames_per_item * rejected_items),
    ):
        methods = {
            "verified-mapping": lambda count, item_count: certify_state(
                item_count, items, 0.05, 0.05
            ),
            "correct-N-r": lambda count, item_count: (
                lambda result: (result.verdict, result.lower, result.upper)
            )(
                partition_robust_verdict(
                    frames,
                    count,
                    frames_per_item,
                    n_items=items,
                    alpha=0.05,
                    delta=0.05,
                )
            ),
            "outer-N60-65-r6": lambda count, item_count: envelope_for_item_counts(
                frames,
                count,
                6,
                range(60, 66),
            ),
            "cap-only-r5": lambda count, item_count: (
                lambda result: (result.verdict, result.lower, result.upper)
            )(
                partition_robust_verdict(
                    frames,
                    count,
                    frames_per_item,
                    alpha=0.05,
                    delta=0.05,
                )
            ),
            "wrong-N65-r5": lambda count, item_count: (
                lambda result: (result.verdict, result.lower, result.upper)
            )(
                partition_robust_verdict(
                    frames,
                    count,
                    frames_per_item,
                    n_items=65,
                    alpha=0.05,
                    delta=0.05,
                )
            ),
            "frame-naive": lambda count, item_count: certify_state(
                count,
                frames,
                0.05,
                0.05,
            ),
        }
        for method, evaluator in methods.items():
            cache = {
                (int(count), int(item_count)): evaluator(
                    int(count), int(item_count)
                )
                for count, item_count in set(
                    zip(frame_alarms.tolist(), rejected_items.tolist())
                )
            }
            evaluated = [
                cache[(int(count), int(item_count))]
                for count, item_count in zip(frame_alarms, rejected_items)
            ]
            states = np.asarray([value[0] for value in evaluated])
            uppers = np.asarray([value[2] for value in evaluated], dtype=float)
            rows.append(
                {
                    "mechanism": mechanism,
                    "method": method,
                    "replicates": replicates,
                    "true_item_fpr": true_rate,
                    **summarize_states(states, uppers, true_rate),
                }
            )
    return rows


def heterogeneity_sensitivity(
    replicates: int,
    rng: np.random.Generator,
) -> list[dict]:
    total_items = 60
    true_mean = 0.06
    rows = []
    for rho in (0.0, 0.01, 0.05, 0.10):
        for batches in (1, 3, 6, 12, 60):
            per_batch = total_items // batches
            if rho == 0.0:
                counts = rng.binomial(total_items, true_mean, size=replicates)
            else:
                concentration = 1.0 / rho - 1.0
                a = true_mean * concentration
                b = (1.0 - true_mean) * concentration
                batch_rates = rng.beta(a, b, size=(replicates, batches))
                counts = rng.binomial(per_batch, batch_rates).sum(axis=1)
            cache = {
                int(count): certify_state(int(count), total_items, 0.05, 0.05)
                for count in np.unique(counts)
            }
            evaluated = [cache[int(count)] for count in counts]
            states = np.asarray([value[0] for value in evaluated])
            uppers = np.asarray([value[2] for value in evaluated], dtype=float)
            rows.append(
                {
                    "intraclass_correlation": rho,
                    "independent_batches": batches,
                    "items_per_batch": per_batch,
                    "total_items": total_items,
                    "replicates": replicates,
                    "true_mean_fpr": true_mean,
                    **summarize_states(states, uppers, true_mean),
                }
            )
    return rows


def vote_geometry() -> list[dict]:
    frames = 55
    items = 11
    maximum = 5
    rows = []
    for alarms in (0, 5, 10, 15, 20, 30, 40):
        for threshold in (1, 2, 3, 4, 5):
            pairs = feasible_item_counts(
                frames,
                alarms,
                maximum,
                n_items=items,
                min_alarms_per_reject=threshold,
            )
            result = partition_robust_verdict(
                frames,
                alarms,
                maximum,
                n_items=items,
                min_alarms_per_reject=threshold,
                alpha=0.05,
                delta=0.05,
            )
            rejected = [pair[0] for pair in pairs]
            rows.append(
                {
                    "frames": frames,
                    "alarms": alarms,
                    "items": items,
                    "max_frames_per_item": maximum,
                    "vote_threshold": threshold,
                    "rejected_min": min(rejected),
                    "rejected_max": max(rejected),
                    "feasible_count": len(pairs),
                    "robust_lower": result.lower,
                    "robust_upper": result.upper,
                    "verdict": result.verdict,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "reliability_extension",
    )
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    dual_rows, dual_summary = dual_gate_rows(
        ROOT / "artifacts" / "fidelity_points.csv"
    )
    metadata_rows = metadata_sensitivity(args.replicates, rng)
    heterogeneity_rows = heterogeneity_sensitivity(args.replicates, rng)
    vote_rows = vote_geometry()

    output = args.output_dir
    write_csv(output / "dual_gate_transistor.csv", dual_rows)
    write_csv(output / "dual_gate_summary.csv", dual_summary)
    write_csv(output / "metadata_sensitivity.csv", metadata_rows)
    write_csv(output / "heterogeneity_sensitivity.csv", heterogeneity_rows)
    write_csv(output / "q_of_r_geometry.csv", vote_rows)
    summary = {
        "seed": args.seed,
        "replicates": args.replicates,
        "dual_gate_summary": dual_summary,
        "metadata_sensitivity": metadata_rows,
        "heterogeneity_sensitivity": heterogeneity_rows,
        "q_of_r_geometry": vote_rows,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
