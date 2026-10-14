#!/usr/bin/env python3
"""Joint false-reject and missed-defect reliability gate.

The component claims are certified with one-sided exact binomial bounds. A
pipeline is release-ready only when both claims are certified. This is an
intersection-union test: no multiplicity correction is required across the two
components to control false release at ``delta``. Multiplicity across candidate
pipelines remains a separate family-wise error problem.
"""

from __future__ import annotations

from dataclasses import dataclass

from cp_state import certify_state


RELEASE_READY = "release-ready"
CONTINUE_TESTING = "continue-testing"
NOT_RELEASE_READY = "not-release-ready"


@dataclass(frozen=True)
class DualRiskGateResult:
    verdict: str
    false_reject_state: str
    false_reject_lower: float
    false_reject_upper: float
    missed_defect_state: str
    missed_defect_lower: float
    missed_defect_upper: float


def dual_risk_gate(
    *,
    false_rejects: int,
    normal_items: int,
    missed_defects: int,
    anomalous_items: int,
    false_reject_budget: float = 0.05,
    missed_defect_budget: float = 0.10,
    delta: float = 0.05,
) -> DualRiskGateResult:
    """Return a three-state release verdict for two reliability requirements."""
    fr_state, fr_lower, fr_upper = certify_state(
        false_rejects,
        normal_items,
        false_reject_budget,
        delta,
    )
    md_state, md_lower, md_upper = certify_state(
        missed_defects,
        anomalous_items,
        missed_defect_budget,
        delta,
    )
    if fr_state == "certified" and md_state == "certified":
        verdict = RELEASE_READY
    elif fr_state == "violated" or md_state == "violated":
        verdict = NOT_RELEASE_READY
    else:
        verdict = CONTINUE_TESTING
    return DualRiskGateResult(
        verdict=verdict,
        false_reject_state=fr_state,
        false_reject_lower=fr_lower,
        false_reject_upper=fr_upper,
        missed_defect_state=md_state,
        missed_defect_lower=md_lower,
        missed_defect_upper=md_upper,
    )
