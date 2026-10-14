#!/usr/bin/env python3
"""Monte Carlo comparison of frame-naive, partition-robust, and mapped audits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))

from cp_state import certify_state
from partition_robust import partition_robust_verdict


def cached_results(
    frame_counts: np.ndarray,
    frames: int,
    items: int,
    maximum: int,
) -> dict[int, tuple[str, float]]:
    return {
        int(alarms): (
            result.verdict,
            result.upper,
        )
        for alarms in np.unique(frame_counts)
        for result in (
            partition_robust_verdict(
                frames,
                int(alarms),
                maximum,
                n_items=items,
                alpha=0.05,
                delta=0.05,
            ),
        )
    }


def cached_fixed_results(
    counts: np.ndarray, sample_size: int
) -> dict[int, tuple[str, float]]:
    return {
        int(count): (state, upper)
        for count in np.unique(counts)
        for state, _, upper in (certify_state(int(count), sample_size),)
    }


def simulate(replicates: int, seed: int) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    items = 60
    frames_per_item = 5
    frames = items * frames_per_item
    rows: list[dict[str, object]] = []
    for mechanism in ("one-frame", "all-frames"):
        for true_fpr in (0.02, 0.05, 0.06, 0.10):
            item_alarms = rng.binomial(items, true_fpr, size=replicates)
            frame_alarms = (
                item_alarms
                if mechanism == "one-frame"
                else frames_per_item * item_alarms
            )
            robust = cached_results(
                frame_alarms, frames, items, frames_per_item
            )
            frame_fixed = cached_fixed_results(frame_alarms, frames)
            item_fixed = cached_fixed_results(item_alarms, items)
            for method in ("frame-naive", "partition-robust", "verified-mapping"):
                lookup = (
                    frame_fixed
                    if method == "frame-naive"
                    else item_fixed
                    if method == "verified-mapping"
                    else robust
                )
                source_counts = (
                    item_alarms if method == "verified-mapping" else frame_alarms
                )
                evaluated = [lookup[int(count)] for count in source_counts]
                states = [value[0] for value in evaluated]
                uppers = [value[1] for value in evaluated]
                states_array = np.asarray(states)
                rows.append(
                    {
                        "mechanism": mechanism,
                        "true_fpr": true_fpr,
                        "method": method,
                        "upper_coverage": float(np.mean(np.asarray(uppers) >= true_fpr)),
                        "certification_rate": float(np.mean(states_array == "certified")),
                        "violation_rate": float(np.mean(states_array == "violated")),
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260726)
    args = parser.parse_args()
    rows = simulate(args.replicates, args.seed)
    print("mechanism,true_fpr,method,upper_coverage,certification_rate,violation_rate")
    for row in rows:
        print(
            f"{row['mechanism']},{row['true_fpr']:.2f},{row['method']},"
            f"{row['upper_coverage']:.5f},{row['certification_rate']:.5f},"
            f"{row['violation_rate']:.5f}"
        )


if __name__ == "__main__":
    main()
