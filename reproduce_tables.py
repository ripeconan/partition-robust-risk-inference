#!/usr/bin/env python3
"""Recompute the manuscript's primary result tables from per-cell counts."""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))

from cp_state import certify_state

DATA_PATH = ROOT / "artifacts" / "fidelity_points.csv"
RULE_VERSION = "cp_one_sided_delta_v1"
TAIL_CONVENTION = "one_sided"
DELTA = 0.05
ALPHAS = {"0.05": 0.05, "0.01": 0.01}

TABLE2_EXPECTED = {
    "ad2": (24, 8, 3),
    "mvtec": (45, 15, 3),
    "visa": (36, 12, 3),
}
TABLE5_EXPECTED_CELLS = 8


class ValidationError(ValueError):
    """Raised when the supplied result table is incomplete or inconsistent."""


def load_rows():
    with DATA_PATH.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_rows(rows):
    errors = []
    invalid_rules = [
        row
        for row in rows
        if row.get("state_rule_version") != RULE_VERSION
        or row.get("tail_convention") != TAIL_CONVENTION
        or float(row["delta"]) != DELTA
    ]
    if invalid_rules:
        errors.append(f"{len(invalid_rules)} rows use an unexpected inference rule")

    keys = [
        (
            row["dataset"],
            row["unit"],
            row["seed"],
            row["model"],
            row["threshold_rule"],
            row["split"],
            row["alpha"],
        )
        for row in rows
    ]
    duplicate_count = len(keys) - len(set(keys))
    if duplicate_count:
        errors.append(f"{duplicate_count} duplicate semantic keys")

    for dataset, (cell_count, unit_count, model_count) in TABLE2_EXPECTED.items():
        for alpha in ALPHAS:
            subset = [
                row
                for row in rows
                if row["dataset"] == dataset
                and row["seed"] == "0"
                and row["threshold_rule"] == "normal_image_quantile"
                and row["split"] == "test_m"
                and row["alpha"] == alpha
            ]
            if len(subset) != cell_count:
                errors.append(
                    f"Table 2 {dataset}, alpha={alpha}: "
                    f"{len(subset)} cells; expected {cell_count}"
                )
            if len({row["unit"] for row in subset}) != unit_count:
                errors.append(
                    f"Table 2 {dataset}, alpha={alpha}: unexpected category count"
                )
            if len({row["model"] for row in subset}) != model_count:
                errors.append(
                    f"Table 2 {dataset}, alpha={alpha}: unexpected model count"
                )

    for rule in ("normal_image_quantile", "ad2_val_pixel_mean_3sd"):
        subset = [
            row
            for row in rows
            if row["dataset"] == "ad2"
            and row["seed"] == "0"
            and row["model"] == "patchcore"
            and row["threshold_rule"] == rule
            and row["split"] == "test_m"
            and row["alpha"] == "0.05"
        ]
        if len(subset) != TABLE5_EXPECTED_CELLS:
            errors.append(
                f"Table 7 {rule}: {len(subset)} cells; "
                f"expected {TABLE5_EXPECTED_CELLS}"
            )

    if errors:
        raise ValidationError("\n".join(errors))


def recompute_verdicts(rows):
    mismatches = 0
    for row in rows:
        verdict, _, _ = certify_state(
            int(row["fp"]),
            int(row["n0"]),
            alpha=ALPHAS[row["alpha"]],
            delta=float(row["delta"]),
        )
        row["_state"] = verdict
        mismatches += verdict != row["state"]
    if mismatches:
        raise ValidationError(
            f"{mismatches} recomputed verdicts differ from the stored values"
        )


def state_counts(rows):
    counts = Counter(row["_state"] for row in rows)
    return (
        counts["certified"],
        counts["violated"],
        counts["unresolved"],
    )


def print_table2(rows):
    print("Table 2: cross-benchmark evidence capacity")
    for dataset in ("ad2", "mvtec", "visa"):
        for alpha in ALPHAS:
            subset = [
                row
                for row in rows
                if row["dataset"] == dataset
                and row["seed"] == "0"
                and row["threshold_rule"] == "normal_image_quantile"
                and row["split"] == "test_m"
                and row["alpha"] == alpha
            ]
            certified, violated, unresolved = state_counts(subset)
            print(
                f"  {dataset:5s} alpha={alpha}: cells={len(subset):3d}; "
                f"C/V/U={certified}/{violated}/{unresolved}"
            )


def print_table5(rows):
    print("Table 7: AD2 PatchCore threshold-rule comparison")
    arms = (
        ("normal_image_quantile", "empirical quantile"),
        ("ad2_val_pixel_mean_3sd", "raw mean+3sigma"),
    )
    for rule, label in arms:
        subset = [
            row
            for row in rows
            if row["dataset"] == "ad2"
            and row["seed"] == "0"
            and row["model"] == "patchcore"
            and row["threshold_rule"] == rule
            and row["split"] == "test_m"
            and row["alpha"] == "0.05"
        ]
        certified, violated, unresolved = state_counts(subset)
        print(
            f"  {label:18s}: cells={len(subset)}; "
            f"C/V/U={certified}/{violated}/{unresolved}"
        )


def main():
    rows = load_rows()
    validate_rows(rows)
    recompute_verdicts(rows)
    print(f"Validated {len(rows)} per-cell rows.\n")
    print_table2(rows)
    print()
    print_table5(rows)


if __name__ == "__main__":
    try:
        main()
    except (KeyError, TypeError, ValueError) as error:
        print(f"Validation failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
