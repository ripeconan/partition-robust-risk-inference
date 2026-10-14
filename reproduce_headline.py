#!/usr/bin/env python3
"""Reproduce the manuscript's headline tables, simulations, and figures."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(label: str, arguments: list[str]) -> None:
    command = [sys.executable, *arguments]
    print(f"\n== {label} ==", flush=True)
    print(" ".join(command), flush=True)
    started = time.perf_counter()
    subprocess.run(command, cwd=ROOT, check=True)
    elapsed = time.perf_counter() - started
    print(f"completed {label} in {elapsed:.1f} s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce table-level and score-level counts, partition and "
            "batch stress tests, reliability extensions, and manuscript figures."
        )
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=100_000,
        help="Monte Carlo replicates (default: 100000, as reported in the paper)",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip the final unit-test suite",
    )
    args = parser.parse_args()
    if args.replicates <= 0:
        parser.error("--replicates must be positive")

    stages: list[tuple[str, list[str]]] = [
        ("table-level evidence", ["reproduce_tables.py"]),
        ("score-level evidence", ["reproduce_from_scores.py"]),
        ("three-seed stability", ["summarize_multiseed.py"]),
        ("partition validation", ["reproduce_partition_analysis.py"]),
        (
            "partition coverage simulation",
            [
                "simulate_partition_coverage.py",
                "--replicates",
                str(args.replicates),
                "--seed",
                "20260726",
            ],
        ),
        (
            "reliability extensions",
            [
                "reliability_extension_analysis.py",
                "--replicates",
                str(args.replicates),
                "--seed",
                "20260802",
            ],
        ),
        ("IJAR uncertainty and recovery analysis", ["ijar_uncertainty_analysis.py"]),
        ("retained baseline figures", ["make_figures.py"]),
        ("IJAR uncertainty figures", ["make_ijar_figures.py"]),
    ]
    if not args.skip_tests:
        stages.append(
            ("unit tests", ["-m", "unittest", "discover", "-s", "tests"])
        )

    started = time.perf_counter()
    for label, arguments in stages:
        run(label, arguments)
    elapsed = time.perf_counter() - started
    print(f"\nAll requested stages completed in {elapsed:.1f} s.")


if __name__ == "__main__":
    main()
