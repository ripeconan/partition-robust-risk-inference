"""Regression tests for exact inference and sample-size calculations."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code"))

from cp_state import certify_state, cp_ci
from partition_robust import feasible_item_counts, partition_robust_verdict
from reliability_gate import dual_risk_gate
from sample_planner import (
    audit_report,
    mde,
    n_add_to_certify,
    n_certify_zero_fp,
)
from reproduce_from_scores import load_records, validate_bundle
from reproduce_partition_analysis import analyze, state_counts


class ExactInferenceTests(unittest.TestCase):
    def test_zero_event_boundaries(self):
        self.assertEqual(n_certify_zero_fp(0.05, 0.05), 59)
        self.assertEqual(n_certify_zero_fp(0.01, 0.05), 299)

    def test_boundary_verdicts(self):
        self.assertEqual(certify_state(0, 58, 0.05, 0.05)[0], "unresolved")
        self.assertEqual(certify_state(0, 59, 0.05, 0.05)[0], "certified")

    def test_additional_items(self):
        self.assertEqual(n_add_to_certify(1, 60, 0.01, 0.05), 413)

    def test_upper_bound_monotonicity(self):
        for n0 in (4, 12, 28, 58, 60):
            with self.subTest(n0=n0):
                self.assertLessEqual(
                    cp_ci(0, n0 + 5, 0.05)[1],
                    cp_ci(0, n0, 0.05)[1],
                )

    def test_mde_decreases_with_sample_size(self):
        self.assertGreater(
            mde(4, 0.05, 0.05, 0.80),
            mde(58, 0.05, 0.05, 0.80),
        )

    def test_invalid_counts(self):
        with self.assertRaises(ValueError):
            cp_ci(0, 0, 0.05)
        with self.assertRaises(ValueError):
            cp_ci(1.5, 10, 0.05)


class FamilyValidationTests(unittest.TestCase):
    @staticmethod
    def _frame(size):
        return pd.DataFrame(
            [
                {
                    "dataset": "example",
                    "unit": f"u{index}",
                    "model": f"m{index % 3}",
                    "seed": 0,
                    "threshold_rule": "raw",
                    "split": "test_m",
                    "alpha": 0.05,
                    "split_hash": f"h{index}",
                    "fp": 0,
                    "n0": 10,
                    "fpr": 0.0,
                }
                for index in range(size)
            ]
        )

    def _audit(self, size):
        with tempfile.NamedTemporaryFile(
            suffix=".csv",
            delete=False,
        ) as handle:
            path = Path(handle.name)
        try:
            self._frame(size).to_csv(path, index=False)
            return audit_report(
                path,
                audit_alphas=[0.05],
                declared_M={"raw": 24},
            )
        finally:
            path.unlink(missing_ok=True)

    def test_complete_family(self):
        audit = self._audit(24)
        self.assertTrue(audit["family_complete"].all())
        self.assertFalse((audit["state_sim"] == "family_incomplete").any())

    def test_incomplete_family(self):
        audit = self._audit(23)
        self.assertFalse(audit["family_complete"].any())
        self.assertTrue((audit["state_sim"] == "family_incomplete").all())


class DualRiskGateTests(unittest.TestCase):
    def test_fpr_only_success_does_not_authorize_release(self):
        result = dual_risk_gate(
            false_rejects=0,
            normal_items=60,
            missed_defects=18,
            anomalous_items=40,
        )
        self.assertEqual(result.false_reject_state, "certified")
        self.assertEqual(result.missed_defect_state, "violated")
        self.assertEqual(result.verdict, "not-release-ready")

    def test_both_component_claims_authorize_release(self):
        result = dual_risk_gate(
            false_rejects=0,
            normal_items=60,
            missed_defects=0,
            anomalous_items=40,
        )
        self.assertEqual(result.verdict, "release-ready")

    def test_one_missed_defect_requires_more_evidence(self):
        result = dual_risk_gate(
            false_rejects=0,
            normal_items=60,
            missed_defects=1,
            anomalous_items=40,
        )
        self.assertEqual(result.false_reject_state, "certified")
        self.assertEqual(result.missed_defect_state, "unresolved")
        self.assertEqual(result.verdict, "continue-testing")


class ReleasedBundleTests(unittest.TestCase):
    def test_production_bundle_is_complete_and_self_consistent(self):
        bundle = ROOT / "artifacts" / "score_bundle"
        records = load_records(bundle / "run_manifest.jsonl")
        manifest = validate_bundle(bundle, records, derived_metric_cells=2414)
        self.assertEqual(len(records), 815)
        self.assertEqual(manifest["runs"], 815)
        self.assertEqual(manifest["derived_metric_cells"], 2414)


class PartitionRobustTests(unittest.TestCase):
    @staticmethod
    def _partitions(size, maximum):
        def build(index, blocks):
            if index == size:
                yield tuple(tuple(block) for block in blocks)
                return
            for block_index, block in enumerate(blocks):
                if len(block) < maximum:
                    block.append(index)
                    yield from build(index + 1, blocks)
                    block.pop()
            blocks.append([index])
            yield from build(index + 1, blocks)
            blocks.pop()

        yield from build(0, [])

    def test_feasible_set_is_sharp_by_exhaustive_enumeration(self):
        for frames in range(1, 8):
            for maximum in range(1, 4):
                for alarms in range(frames + 1):
                    observed = set()
                    alarmed = set(range(alarms))
                    for partition in self._partitions(frames, maximum):
                        rejected = sum(
                            bool(alarmed.intersection(block)) for block in partition
                        )
                        observed.add((rejected, len(partition)))
                    with self.subTest(
                        frames=frames, alarms=alarms, maximum=maximum
                    ):
                        self.assertEqual(
                            set(feasible_item_counts(frames, alarms, maximum)),
                            observed,
                        )

    def test_q_of_r_feasible_set_is_sharp_by_dynamic_enumeration(self):
        for maximum in range(2, 6):
            for threshold in range(2, maximum + 1):
                for items in range(1, 6):
                    states = {(0, 0, 0)}
                    for _ in range(items):
                        next_states = set()
                        for used_frames, used_alarms, rejected in states:
                            for size in range(1, maximum + 1):
                                for item_alarms in range(size + 1):
                                    next_states.add(
                                        (
                                            used_frames + size,
                                            used_alarms + item_alarms,
                                            rejected + (item_alarms >= threshold),
                                        )
                                    )
                        states = next_states
                    observed = {}
                    for frames, alarms, rejected in states:
                        observed.setdefault((frames, alarms), set()).add(rejected)
                    for (frames, alarms), rejected_counts in observed.items():
                        with self.subTest(
                            frames=frames,
                            alarms=alarms,
                            items=items,
                            maximum=maximum,
                            threshold=threshold,
                        ):
                            computed = feasible_item_counts(
                                frames,
                                alarms,
                                maximum,
                                n_items=items,
                                min_alarms_per_reject=threshold,
                            )
                            self.assertEqual(
                                {rejected for rejected, _ in computed},
                                rejected_counts,
                            )

    def test_q_of_r_reduces_uncertainty_relative_to_any_frame(self):
        any_frame = feasible_item_counts(55, 10, 5, n_items=11)
        majority = feasible_item_counts(
            55,
            10,
            5,
            n_items=11,
            min_alarms_per_reject=3,
        )
        self.assertEqual(any_frame[0], (2, 11))
        self.assertEqual(any_frame[-1], (10, 11))
        self.assertEqual(majority, ((0, 11), (1, 11), (2, 11), (3, 11)))

    def test_invalid_vote_threshold(self):
        with self.assertRaises(ValueError):
            feasible_item_counts(
                20,
                2,
                5,
                min_alarms_per_reject=6,
            )

    def test_known_item_count_contains_observed_ad2_grouping(self):
        pairs = feasible_item_counts(55, 4, 5, n_items=11)
        self.assertIn((3, 11), pairs)
        self.assertEqual(pairs, ((1, 11), (2, 11), (3, 11), (4, 11)))

    def test_high_alarm_rate_is_robustly_violated(self):
        result = partition_robust_verdict(
            60, 20, 5, n_items=12, alpha=0.05, delta=0.05
        )
        self.assertEqual(result.verdict, "violated")

    def test_zero_alarm_capacity_with_unknown_item_count(self):
        self.assertEqual(
            partition_robust_verdict(295, 0, 5).verdict,
            "certified",
        )
        self.assertEqual(
            partition_robust_verdict(290, 0, 5).verdict,
            "unresolved",
        )

    def test_invalid_item_count(self):
        with self.assertRaises(ValueError):
            feasible_item_counts(20, 2, 5, n_items=3)

    def test_ad2_multiseed_counts(self):
        result = analyze()
        self.assertEqual(len(result), 120)
        self.assertEqual(state_counts(result, "frame_state"), "0/79/41")
        self.assertEqual(state_counts(result, "cap_only_state"), "0/40/80")
        self.assertEqual(
            state_counts(result, "partition_robust_state"), "0/64/56"
        )
        self.assertEqual(state_counts(result, "item_state"), "0/84/36")


if __name__ == "__main__":
    unittest.main()
