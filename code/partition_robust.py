#!/usr/bin/env python3
"""Exact item-level FPR auditing when frame-to-item membership is incomplete."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass

from cp_state import (
    STATE_CERTIFIED,
    STATE_UNRESOLVED,
    STATE_VIOLATED,
    _as_int,
    _probability,
    cp_ci,
)


@dataclass(frozen=True)
class PartitionRobustResult:
    """Envelope of exact bounds over every grouping compatible with metadata."""

    verdict: str
    lower: float
    upper: float
    lower_pair: tuple[int, int]
    upper_pair: tuple[int, int]
    feasible_pairs: tuple[tuple[int, int], ...]


def feasible_item_counts(
    frames: int,
    alarms: int,
    max_frames_per_item: int,
    n_items: int | None = None,
    min_alarms_per_reject: int = 1,
) -> tuple[tuple[int, int], ...]:
    """Return all feasible ``(rejected_items, items)`` pairs.

    Each item contains between one and ``max_frames_per_item`` frames and is
    rejected when at least ``min_alarms_per_reject`` constituent frames alarm.
    ``n_items`` may be supplied when the item count is known but the
    frame-to-item mapping is unavailable. The returned set is sharp: every
    pair is attained by at least one grouping. The default threshold of one
    recovers any-frame aggregation.
    """
    frames = _as_int(frames, "frames")
    alarms = _as_int(alarms, "alarms")
    max_frames_per_item = _as_int(
        max_frames_per_item, "max_frames_per_item"
    )
    min_alarms_per_reject = _as_int(
        min_alarms_per_reject, "min_alarms_per_reject"
    )
    if frames < 1:
        raise ValueError("frames must be positive")
    if not 0 <= alarms <= frames:
        raise ValueError("alarms must satisfy 0 <= alarms <= frames")
    if max_frames_per_item < 1:
        raise ValueError("max_frames_per_item must be positive")
    if not 1 <= min_alarms_per_reject <= max_frames_per_item:
        raise ValueError(
            "min_alarms_per_reject must lie between one and "
            "max_frames_per_item"
        )

    minimum_items = math.ceil(frames / max_frames_per_item)
    if n_items is None:
        item_counts = range(minimum_items, frames + 1)
    else:
        n_items = _as_int(n_items, "n_items")
        if not minimum_items <= n_items <= frames:
            raise ValueError(
                "n_items is incompatible with the frame count and cluster-size bound"
            )
        item_counts = (n_items,)

    feasible: list[tuple[int, int]] = []
    normal_frames = frames - alarms
    threshold = min_alarms_per_reject
    for items in item_counts:
        if threshold == 1:
            if alarms == 0:
                feasible.append((0, items))
                continue
            rejected_min = max(
                1,
                math.ceil(alarms / max_frames_per_item),
                items - normal_frames,
            )
            rejected_max = min(alarms, items)
        else:
            # A rejected item consumes at least q alarms. Non-rejected items
            # can absorb at most q-1 alarms; rejected items can absorb at most
            # r. These constraints give the lower limit. In addition to the
            # alarm budget, K rejected items require at least N+(q-1)K total
            # frames, which gives the second upper limit. The interval is sharp.
            q = threshold
            excess_alarms = alarms - (q - 1) * items
            denominator = max_frames_per_item - q + 1
            rejected_min = max(
                0,
                (excess_alarms + denominator - 1) // denominator,
            )
            rejected_max = min(
                items,
                alarms // q,
                (frames - items) // (q - 1),
            )
        feasible.extend(
            (rejected, items)
            for rejected in range(rejected_min, rejected_max + 1)
        )
    if not feasible:
        raise ValueError("no grouping is compatible with the supplied constraints")
    return tuple(feasible)


def partition_robust_verdict(
    frames: int,
    alarms: int,
    max_frames_per_item: int,
    *,
    n_items: int | None = None,
    min_alarms_per_reject: int = 1,
    alpha: float = 0.05,
    delta: float = 0.05,
) -> PartitionRobustResult:
    """Compute a three-state verdict uniform over all compatible groupings."""
    alpha = _probability(alpha, "alpha")
    delta = _probability(delta, "delta")
    pairs = feasible_item_counts(
        frames,
        alarms,
        max_frames_per_item,
        n_items=n_items,
        min_alarms_per_reject=min_alarms_per_reject,
    )
    bounds = [(rejected, items, *cp_ci(rejected, items, delta)) for rejected, items in pairs]
    lower_row = min(bounds, key=lambda row: row[2])
    upper_row = max(bounds, key=lambda row: row[3])
    robust_lower = lower_row[2]
    robust_upper = upper_row[3]
    if robust_upper <= alpha:
        verdict = STATE_CERTIFIED
    elif robust_lower > alpha:
        verdict = STATE_VIOLATED
    else:
        verdict = STATE_UNRESOLVED
    return PartitionRobustResult(
        verdict=verdict,
        lower=robust_lower,
        upper=robust_upper,
        lower_pair=(lower_row[0], lower_row[1]),
        upper_pair=(upper_row[0], upper_row[1]),
        feasible_pairs=pairs,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exact item-FPR audit over every compatible frame partition."
    )
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--alarms", type=int, required=True)
    parser.add_argument("--max-frames-per-item", type=int, required=True)
    parser.add_argument("--min-alarms-per-reject", type=int, default=1)
    parser.add_argument("--n-items", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--delta", type=float, default=0.05)
    args = parser.parse_args()
    result = partition_robust_verdict(
        args.frames,
        args.alarms,
        args.max_frames_per_item,
        n_items=args.n_items,
        min_alarms_per_reject=args.min_alarms_per_reject,
        alpha=args.alpha,
        delta=args.delta,
    )
    print(
        json.dumps(
            {
                "verdict": result.verdict,
                "lower": result.lower,
                "upper": result.upper,
                "lower_pair": result.lower_pair,
                "upper_pair": result.upper_pair,
                "feasible_pair_count": len(result.feasible_pairs),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
