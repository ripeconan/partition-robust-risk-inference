#!/usr/bin/env python3
"""Summarize the three-dataset primary matrix across seeds 0, 1, and 2."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from reproduce_from_scores import derive, load_records, validate_bundle


ROOT = Path(__file__).resolve().parent
EXPECTED_CELLS = {"ad2": 24, "mvtec": 45, "visa": 36}
STATE_CODE = {"certified": "C", "violated": "V", "unresolved": "U"}


def select_primary(rows: list[dict]) -> list[dict]:
    return [
        row
        for row in rows
        if row["threshold_rule"] == "normal_image_quantile"
        and row["split"] == "test_m"
        and row["seed"] in (0, 1, 2)
    ]


def summarize(rows: list[dict]) -> bool:
    selected = select_primary(rows)
    complete = True
    for alpha in (0.05, 0.01):
        print(f"\nalpha={alpha:.2f}")
        for dataset, expected in EXPECTED_CELLS.items():
            dataset_rows = [
                row
                for row in selected
                if row["dataset"] == dataset and row["alpha"] == alpha
            ]
            by_seed: dict[int, dict[tuple[str, str], str]] = {}
            for seed in (0, 1, 2):
                seed_rows = [row for row in dataset_rows if row["seed"] == seed]
                counts = Counter(row["state"] for row in seed_rows)
                status = "complete" if len(seed_rows) == expected else "INCOMPLETE"
                complete &= len(seed_rows) == expected
                print(
                    f"  {dataset:5s} seed={seed}: cells={len(seed_rows):2d}/{expected}; "
                    f"C/V/U={counts['certified']}/{counts['violated']}/"
                    f"{counts['unresolved']} [{status}]"
                )
                by_seed[seed] = {
                    (row["unit_name"], row["model"]): row["state"]
                    for row in seed_rows
                }

            key_sets = [set(by_seed[seed]) for seed in (0, 1, 2)]
            if not all(keys == key_sets[0] for keys in key_sets[1:]):
                print("    stability: unavailable until all seeds share the same cells")
                complete = False
                continue
            patterns = Counter(
                "".join(STATE_CODE[by_seed[seed][key]] for seed in (0, 1, 2))
                for key in sorted(key_sets[0])
            )
            stable = sum(count for pattern, count in patterns.items() if len(set(pattern)) == 1)
            pattern_text = ", ".join(
                f"{pattern}:{count}" for pattern, count in patterns.most_common()
            )
            print(f"    stable={stable}/{expected}; patterns={pattern_text}")
    return complete


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=ROOT / "artifacts" / "score_bundle",
    )
    args = parser.parse_args()
    records = load_records(args.bundle_dir / "run_manifest.jsonl")
    validate_bundle(args.bundle_dir, records)
    with np.load(args.bundle_dir / "run_scores.npz", allow_pickle=False) as archive:
        rows = derive(records, archive)
    validate_bundle(args.bundle_dir, records, len(rows))
    print(f"Validated {len(records)} score runs and {len(rows)} derived metric cells.")
    if not summarize(rows):
        raise SystemExit("primary multi-seed matrix is incomplete")


if __name__ == "__main__":
    main()
