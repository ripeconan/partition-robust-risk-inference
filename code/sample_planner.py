#!/usr/bin/env python3
"""Sample-size and power calculations for item-level FPR claims."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import binom

from cp_state import cp_ci, certify_state

REQUIRED_COLUMNS = (
    "dataset",
    "unit",
    "model",
    "seed",
    "threshold_rule",
    "split",
    "alpha",
    "split_hash",
    "fp",
    "n0",
    "fpr",
)
SEMANTIC_KEY = (
    "dataset",
    "unit",
    "model",
    "seed",
    "threshold_rule",
    "split",
    "alpha",
)


def _cp_upper(k, n, delta):
    return cp_ci(k, n, delta)[1]


def n_certify_zero_fp(alpha=0.05, delta=0.05):
    """Minimum item count that can certify after zero false positives."""
    if not (0 < alpha < 1 and 0 < delta < 1):
        raise ValueError("alpha and delta must be in (0, 1)")
    return int(math.ceil(math.log(delta) / math.log1p(-alpha)))


def n_certify_tolerance(alpha=0.05, delta=0.05, max_fp=0):
    """Minimum item count that can certify with at most ``max_fp`` events."""
    if max_fp < 0:
        raise ValueError("max_fp must be nonnegative")
    high = max(max_fp + 1, n_certify_zero_fp(alpha, delta))
    while _cp_upper(max_fp, high, delta) > alpha:
        high *= 2
        if high > 10**8:
            raise RuntimeError("search exceeded 100 million items")

    low = max_fp + 1
    while low < high:
        midpoint = (low + high) // 2
        if _cp_upper(max_fp, midpoint, delta) <= alpha:
            high = midpoint
        else:
            low = midpoint + 1
    return int(low)


def n_add_to_certify(fp, n0, alpha=0.05, delta=0.05):
    """Additional zero-event items required to reach certification.

    This is an optimistic planning calculation: ``fp`` remains fixed while the
    total number of independent items increases.
    """
    verdict, _, _ = certify_state(fp, n0, alpha, delta)
    if verdict == "certified":
        return 0

    high = max(n0 + 1, n_certify_zero_fp(alpha, delta) + fp * 20)
    while _cp_upper(fp, high, delta) > alpha:
        high *= 2
        if high > 10**9:
            raise RuntimeError("search exceeded one billion items")

    low = n0
    while low < high:
        midpoint = (low + high) // 2
        if _cp_upper(fp, midpoint, delta) <= alpha:
            high = midpoint
        else:
            low = midpoint + 1
    return int(low) - n0


def mde(n0, alpha=0.05, delta=0.05, power=0.80):
    """Minimum detectable true-FPR excess for the violation test."""
    if n0 <= 0:
        raise ValueError("n0 must be positive")
    if not 0 < power < 1:
        raise ValueError("power must be in (0, 1)")

    critical_count = next(
        (k for k in range(n0 + 1) if cp_ci(k, n0, delta)[0] > alpha),
        None,
    )
    if critical_count is None:
        return float("inf")

    def power_difference(probability):
        return float(binom.sf(critical_count - 1, n0, probability)) - power

    try:
        detectable_rate = brentq(
            power_difference,
            alpha,
            1.0 - 1e-9,
            xtol=1e-7,
        )
    except ValueError:
        return float("inf")
    return float(detectable_rate - alpha)


def cluster_design_effect(n_observations, mean_cluster_size, icc):
    """Effective count under an exchangeable-correlation planning model.

    This quantity is for sensitivity analysis and is not used as the sample
    count in exact binomial inference.
    """
    if mean_cluster_size <= 1 or icc <= 0:
        return float(n_observations)
    return float(n_observations) / (1.0 + (mean_cluster_size - 1.0) * icc)


def supported_claims(
    fp,
    n0,
    alpha_targets=(0.01, 0.05, 0.10),
    delta=0.05,
):
    """Return the three-state verdict at each requested FPR budget."""
    return {alpha: certify_state(fp, n0, alpha, delta)[0] for alpha in alpha_targets}


def _validate_input(frame):
    missing = [column for column in REQUIRED_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    for column in ("fp", "n0", "seed"):
        numeric = pd.to_numeric(frame[column], errors="raise")
        if not np.equal(numeric, np.floor(numeric)).all():
            raise ValueError(f"{column} contains non-integer values")
        frame[column] = numeric.astype(int)

    invalid = frame[(frame["fp"] < 0) | (frame["n0"] < 1) | (frame["fp"] > frame["n0"])]
    if not invalid.empty:
        records = invalid[list(SEMANTIC_KEY)].head().to_dict("records")
        raise ValueError(f"invalid fp/n0 rows: {records}")


def audit_report(
    csv_path,
    audit_alphas=(0.05, 0.01),
    delta=0.05,
    power=0.80,
    seed0_only=True,
    lock_alpha0=0.05,
    verification_split="test_m",
    declared_M=None,
):
    """Calculate marginal and Bonferroni-adjusted per-cell audit results.

    ``declared_M`` maps each threshold rule to its pre-declared family size. A
    family with fewer observed cells is marked ``family_incomplete``.
    """
    frame = pd.read_csv(csv_path)
    _validate_input(frame)

    if seed0_only:
        frame = frame[frame["seed"] == 0].copy()
    frame = frame[frame["split"] == verification_split].copy()
    frame = frame[np.isclose(frame["alpha"].astype(float), lock_alpha0)].copy()

    duplicates = frame.duplicated(subset=list(SEMANTIC_KEY), keep=False)
    if duplicates.any():
        rows = frame.loc[duplicates, list(SEMANTIC_KEY)].head()
        raise ValueError(f"duplicate semantic keys:\n{rows}")

    has_source_alpha = "source_alpha" in frame
    records = []
    for _, row in frame.iterrows():
        fp = int(row["fp"])
        n0 = int(row["n0"])
        source_alpha = None
        if has_source_alpha and not pd.isna(row["source_alpha"]):
            source_alpha = float(row["source_alpha"])

        for alpha in audit_alphas:
            verdict, lower, upper = certify_state(fp, n0, alpha, delta)
            records.append(
                {
                    "dataset": row["dataset"],
                    "unit": row["unit"],
                    "seed": int(row["seed"]),
                    "model": row["model"],
                    "threshold_rule": row["threshold_rule"],
                    "split": row["split"],
                    "split_hash": row["split_hash"],
                    "source_alpha": source_alpha,
                    "metric_alpha": float(row["alpha"]),
                    "audit_alpha": alpha,
                    "fp": fp,
                    "n0": n0,
                    "fpr": float(row["fpr"]),
                    "fpr_lower": lower,
                    "fpr_upper": upper,
                    "state": verdict,
                    "n_add_if_no_more_fp": n_add_to_certify(fp, n0, alpha, delta),
                    "mde": mde(n0, alpha, delta, power),
                }
            )

    audit = pd.DataFrame(records)
    family_columns = ["threshold_rule", "audit_alpha"]
    observed = audit.groupby(family_columns)["dataset"].transform("count").astype(int)

    if declared_M:
        declared = audit["threshold_rule"].map(declared_M)
        audit["M_declared"] = declared
        audit["M_family"] = declared.fillna(observed).astype(int)
        audit["family_complete"] = declared.notna() & (observed == audit["M_family"])
    else:
        audit["M_declared"] = pd.NA
        audit["M_family"] = observed
        audit["family_complete"] = pd.NA

    simultaneous = []
    for _, row in audit.iterrows():
        complete = row["family_complete"]
        if not pd.isna(complete) and not bool(complete):
            simultaneous.append(("family_incomplete", float("nan"), float("nan")))
            continue
        simultaneous.append(
            certify_state(
                row["fp"],
                row["n0"],
                row["audit_alpha"],
                delta / int(row["M_family"]),
            )
        )

    simultaneous_frame = pd.DataFrame(
        simultaneous,
        columns=["state_sim", "fpr_lower_sim", "fpr_upper_sim"],
        index=audit.index,
    )
    return pd.concat([audit, simultaneous_frame], axis=1)


def summarize(audit):
    """Summarize audit outcomes by dataset, threshold rule, and FPR budget."""
    records = []
    groups = audit.groupby(["dataset", "threshold_rule", "audit_alpha"])
    for (dataset, rule, alpha), group in groups:
        family_complete = group["family_complete"].iloc[0]
        records.append(
            {
                "dataset": dataset,
                "threshold_rule": rule,
                "audit_alpha": alpha,
                "M_family": int(group["M_family"].iloc[0]),
                "n_cells": len(group),
                "certified": int((group["state"] == "certified").sum()),
                "violated": int((group["state"] == "violated").sum()),
                "unresolved": int((group["state"] == "unresolved").sum()),
                "certified_sim": int((group["state_sim"] == "certified").sum()),
                "violated_sim": int((group["state_sim"] == "violated").sum()),
                "unresolved_sim": int((group["state_sim"] == "unresolved").sum()),
                "incomplete_sim": int(
                    (group["state_sim"] == "family_incomplete").sum()
                ),
                "family_complete": (
                    None if pd.isna(family_complete) else bool(family_complete)
                ),
                "median_n0": float(group["n0"].median()),
                "median_mde": float(group["mde"].median()),
                "median_n_add_if_no_more_fp": float(
                    group["n_add_if_no_more_fp"].median()
                ),
            }
        )
    return pd.DataFrame(records)


def parse_args():
    default_csv = (
        Path(__file__).resolve().parents[1] / "artifacts" / "fidelity_points.csv"
    )
    parser = argparse.ArgumentParser(
        description="Calculate sample-size and power quantities for FPR claims."
    )
    parser.add_argument("--csv", type=Path, default=default_csv)
    parser.add_argument(
        "--audit-alphas",
        type=float,
        nargs="+",
        default=[0.05, 0.01],
    )
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--power", type=float, default=0.80)
    parser.add_argument("--lock-alpha0", type=float, default=0.05)
    parser.add_argument("--verification-split", default="test_m")
    parser.add_argument("--all-seeds", action="store_true")
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    audit = audit_report(
        args.csv,
        args.audit_alphas,
        args.delta,
        args.power,
        seed0_only=not args.all_seeds,
        lock_alpha0=args.lock_alpha0,
        verification_split=args.verification_split,
    )
    print(
        "Zero-event certification floors: "
        f"5% FPR = {n_certify_zero_fp(0.05, args.delta)} items; "
        f"1% FPR = {n_certify_zero_fp(0.01, args.delta)} items"
    )
    print()
    print(summarize(audit).to_string(index=False))

    if args.out:
        audit.to_csv(args.out, index=False)
        print(f"\nPer-cell results written to {args.out}")


if __name__ == "__main__":
    main()
