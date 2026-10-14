#!/usr/bin/env python3
"""Reproduce the partition-robust AD2 Test-S sampling-unit analysis."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))

from cp_state import certify_state
from partition_robust import partition_robust_verdict


INPUT = ROOT / "artifacts" / "ad2_partition_observations.csv"


def analyze(path: Path = INPUT) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = {
        "unit",
        "model",
        "seed",
        "n_frame",
        "k_frame",
        "n_item",
        "k_item",
        "frame_state",
        "item_state",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if data.duplicated(["unit", "model", "seed"]).any():
        raise ValueError("duplicate unit/model/seed rows")

    rows = []
    for row in data.itertuples(index=False):
        frame_state = certify_state(row.k_frame, row.n_frame, 0.05, 0.05)[0]
        item_state = certify_state(row.k_item, row.n_item, 0.05, 0.05)[0]
        if frame_state != row.frame_state or item_state != row.item_state:
            raise ValueError(
                f"stored verdict mismatch for {row.unit}/{row.model}/seed{row.seed}"
            )
        robust = partition_robust_verdict(
            row.n_frame,
            row.k_frame,
            5,
            n_items=row.n_item,
            alpha=0.05,
            delta=0.05,
        )
        cap_only = partition_robust_verdict(
            row.n_frame,
            row.k_frame,
            5,
            alpha=0.05,
            delta=0.05,
        )
        rows.append(
            {
                **row._asdict(),
                "cap_only_state": cap_only.verdict,
                "cap_only_lower": cap_only.lower,
                "cap_only_upper": cap_only.upper,
                "partition_robust_state": robust.verdict,
                "partition_robust_lower": robust.lower,
                "partition_robust_upper": robust.upper,
                "lower_pair": f"{robust.lower_pair[0]}/{robust.lower_pair[1]}",
                "upper_pair": f"{robust.upper_pair[0]}/{robust.upper_pair[1]}",
            }
        )
    return pd.DataFrame(rows)


def state_counts(data: pd.DataFrame, column: str) -> str:
    counts = data[column].value_counts()
    return "/".join(str(int(counts.get(state, 0))) for state in (
        "certified",
        "violated",
        "unresolved",
    ))


def main() -> None:
    result = analyze()
    print("AD2 Test-S sampling-unit evidence at alpha=delta=5%")
    print("C/V/U denotes certified/violated/unresolved.\n")
    print("seed  frame-naive  cap-only robust  count+cap robust  verified mapping")
    for seed, selected in result.groupby("seed", sort=True):
        print(
            f"{seed:>4}  {state_counts(selected, 'frame_state'):>11}  "
            f"{state_counts(selected, 'cap_only_state'):>15}  "
            f"{state_counts(selected, 'partition_robust_state'):>16}  "
            f"{state_counts(selected, 'item_state'):>16}"
        )
    print(
        f" all  {state_counts(result, 'frame_state'):>11}  "
        f"{state_counts(result, 'cap_only_state'):>15}  "
        f"{state_counts(result, 'partition_robust_state'):>16}  "
        f"{state_counts(result, 'item_state'):>16}"
    )


if __name__ == "__main__":
    main()
