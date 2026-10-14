#!/usr/bin/env python3
"""Recompute per-run FPR evidence directly from the released score bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))

from cp_state import certify_state


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(int)
    positive = labels == 1
    negative = labels == 0
    if not positive.any() or not negative.any():
        return float("nan")
    ranks = rankdata(scores, method="average")
    numerator = ranks[positive].sum() - positive.sum() * (positive.sum() + 1) / 2
    return float(numerator / (positive.sum() * negative.sum()))


def load_records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    keys = [tuple(record["semantic_key"]) for record in records]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate semantic keys in run manifest")
    return records


def validate_bundle(
    bundle_dir: Path,
    records: list[dict],
    derived_metric_cells: int | None = None,
) -> dict:
    manifest_path = bundle_dir / "score_bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("runs") != len(records):
        raise ValueError("score bundle run count does not match its manifest")
    for name, expected in manifest.get("files", {}).items():
        path = bundle_dir / name
        if not path.is_file():
            raise ValueError(f"score bundle file is missing: {name}")
        if path.stat().st_size != expected.get("bytes"):
            raise ValueError(f"score bundle file size mismatch: {name}")
        if sha256(path) != expected.get("sha256"):
            raise ValueError(f"score bundle checksum mismatch: {name}")
    if derived_metric_cells is not None:
        if manifest.get("derived_metric_cells") != derived_metric_cells:
            raise ValueError(
                "derived metric-cell count does not match the score bundle manifest"
            )
    return manifest


def arrays_for_run(archive: np.lib.npyio.NpzFile, run_id: str) -> dict[str, np.ndarray]:
    prefix = run_id + "__"
    return {
        key[len(prefix):]: archive[key]
        for key in archive.files
        if key.startswith(prefix)
    }


def threshold_and_scores(
    summary: dict,
    arrays: dict[str, np.ndarray],
    split_suffix: str,
) -> tuple[float, np.ndarray]:
    rule = summary["threshold_rule"]
    if rule == "normal_image_quantile":
        definition = summary.get("threshold_definition") or {}
        source_alpha = definition.get("source_alpha", 0.05)
        if isinstance(source_alpha, list):
            if 0.05 not in source_alpha:
                raise ValueError("empirical run does not contain the locked 5% quantile")
            source_alpha = 0.05
        if source_alpha is None:
            source_alpha = 0.05
        tau = float(np.quantile(arrays["cal_scores"].astype(float), 1 - source_alpha))
        return tau, arrays[f"scores_{split_suffix}"].astype(float)
    statistics = summary["threshold_statistics"]
    if rule == "ad2_val_pixel_mean_3sd":
        return float(statistics["tau_seg"]), arrays[
            f"max_map_scores_{split_suffix}"
        ].astype(float)
    if rule == "ad2_val_pixel_mean_3sd_norm":
        return float(statistics["tau_seg_norm"]), arrays[
            f"max_map_scores_{split_suffix}_normalized"
        ].astype(float)
    raise ValueError(f"unsupported threshold rule: {rule}")


def item_ids(paths: np.ndarray) -> np.ndarray:
    pattern = re.compile(r"_(regular|overexposed|underexposed|shift_[123])$")
    return np.asarray([pattern.sub("", Path(path).stem) for path in paths.astype(str)])


def aggregate_ad2_items(
    scores: np.ndarray,
    labels: np.ndarray,
    paths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    groups: dict[tuple[int, str], list[int]] = {}
    for index, (label, item) in enumerate(
        zip(labels.astype(int), item_ids(paths), strict=True)
    ):
        groups.setdefault((int(label), str(item)), []).append(index)
    item_labels = []
    item_scores = []
    for (label, _), indices in sorted(groups.items()):
        item_labels.append(label)
        item_scores.append(float(np.max(scores[indices])))
    return np.asarray(item_labels, dtype=int), np.asarray(item_scores, dtype=float)


def derive(records: list[dict], archive: np.lib.npyio.NpzFile) -> list[dict]:
    rows = []
    for record in records:
        summary = record["summary"]
        arrays = arrays_for_run(archive, record["run_id"])
        for split, suffix in (("test_m", "m"), ("test_s", "s")):
            labels = arrays[f"labels_{suffix}"].astype(int)
            if not len(labels):
                continue
            paths = arrays[f"paths_{suffix}"].astype(str)
            tau, scores = threshold_and_scores(summary, arrays, suffix)
            unit = "frame"
            if summary["dataset"] == "ad2":
                labels, scores = aggregate_ad2_items(scores, labels, paths)
                unit = "physical_item"
            alarms = scores > tau
            for alpha in (0.01, 0.05):
                normal = labels == 0
                anomaly = labels == 1
                fp = int(alarms[normal].sum())
                n0 = int(normal.sum())
                tp = int(alarms[anomaly].sum())
                n1 = int(anomaly.sum())
                state, lower, upper = certify_state(fp, n0, alpha, 0.05)
                inference_eligible = (
                    summary["threshold_rule"] != "ad2_val_pixel_mean_3sd_norm"
                )
                if not inference_eligible:
                    state = "descriptive_only"
                rows.append(
                    {
                        "dataset": summary["dataset"],
                        "unit_name": summary["unit"],
                        "model": summary["model"],
                        "seed": int(summary["seed"]),
                        "threshold_rule": summary["threshold_rule"],
                        "split": split,
                        "sampling_unit": unit,
                        "alpha": alpha,
                        "tau": tau,
                        "fp": fp,
                        "n0": n0,
                        "fpr": fp / n0,
                        "fpr_lower": lower,
                        "fpr_upper": upper,
                        "state": state,
                        "inference_eligible": inference_eligible,
                        "tp": tp,
                        "n1": n1,
                        "tpr": tp / n1 if n1 else float("nan"),
                        "auroc": binary_auroc(labels, scores),
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def headline(rows: list[dict]) -> None:
    selected = [
        row
        for row in rows
        if row["seed"] == 0
        and row["threshold_rule"] == "normal_image_quantile"
        and row["split"] == "test_m"
    ]
    print("Score-derived primary seed-0 Test-M evidence")
    for dataset in ("ad2", "mvtec", "visa"):
        for alpha in (0.05, 0.01):
            cells = [
                row
                for row in selected
                if row["dataset"] == dataset and row["alpha"] == alpha
            ]
            counts = Counter(row["state"] for row in cells)
            print(
                f"  {dataset:5s} alpha={alpha:.2f}: cells={len(cells):3d}; "
                f"C/V/U={counts['certified']}/{counts['violated']}/{counts['unresolved']}"
            )
    print("\nMatched AD2 PatchCore seed-0 Test-M threshold transport")
    for rule in ("normal_image_quantile", "ad2_val_pixel_mean_3sd"):
        cells = [
            row
            for row in rows
            if row["dataset"] == "ad2"
            and row["model"] == "patchcore"
            and row["seed"] == 0
            and row["split"] == "test_m"
            and row["alpha"] == 0.05
            and row["threshold_rule"] == rule
        ]
        counts = Counter(row["state"] for row in cells)
        print(
            f"  {rule:28s}: cells={len(cells):2d}; "
            f"C/V/U={counts['certified']}/{counts['violated']}/{counts['unresolved']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle-dir", type=Path, default=ROOT / "artifacts" / "score_bundle"
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    records = load_records(args.bundle_dir / "run_manifest.jsonl")
    validate_bundle(args.bundle_dir, records)
    with np.load(args.bundle_dir / "run_scores.npz", allow_pickle=False) as archive:
        rows = derive(records, archive)
    validate_bundle(args.bundle_dir, records, len(rows))
    print(f"Validated {len(records)} score runs and {len(rows)} derived metric cells.")
    headline(rows)
    if args.output:
        write_csv(args.output, rows)
        print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
