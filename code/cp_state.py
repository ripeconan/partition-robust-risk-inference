#!/usr/bin/env python3
"""Exact binomial intervals and three-state low-FPR verdicts."""

from __future__ import annotations

import math
import numbers
import re
from typing import Tuple

from scipy.stats import beta

STATE_CERTIFIED = "certified"
STATE_VIOLATED = "violated"
STATE_UNRESOLVED = "unresolved"


def _as_int(value, name: str) -> int:
    """Return an integer count without silently truncating non-integers."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer count; got {value!r}")
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if re.fullmatch(r"[+-]?\d+", text):
            return int(text)
        raise ValueError(f"{name} must be an integer count; got {value!r}")
    if isinstance(value, numbers.Real):
        numeric = float(value)
        if math.isfinite(numeric) and numeric.is_integer():
            return int(numeric)
    raise ValueError(f"{name} must be an integer count; got {value!r}")


def _probability(value, name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric; got {value!r}") from exc
    if not (math.isfinite(numeric) and 0 < numeric < 1):
        raise ValueError(f"{name} must be finite and in (0, 1); got {value!r}")
    return numeric


def cp_ci(k: int, n: int, delta: float = 0.05) -> Tuple[float, float]:
    """Return one-sided Clopper--Pearson lower and upper bounds.

    Each bound has error level ``delta``. The boundary values are zero when
    ``k == 0`` and one when ``k == n``.
    """
    k = _as_int(k, "k")
    n = _as_int(n, "n")
    delta = _probability(delta, "delta")
    if not (0 <= k <= n and n >= 1):
        raise ValueError(
            f"counts must satisfy 0 <= k <= n and n >= 1; got k={k}, n={n}"
        )

    lower = float(beta.ppf(delta, k, n - k + 1)) if k > 0 else 0.0
    upper = float(beta.ppf(1 - delta, k + 1, n - k)) if k < n else 1.0
    return lower, upper


def certify_state(fp: int, n0: int, alpha: float = 0.05, delta: float = 0.05):
    """Return ``(verdict, lower, upper)`` for an item-level FPR claim.

    The claim is certified when the one-sided upper bound is at most ``alpha``,
    violated when the one-sided lower bound exceeds ``alpha``, and unresolved
    otherwise.
    """
    alpha = _probability(alpha, "alpha")
    lower, upper = cp_ci(fp, n0, delta)
    if upper <= alpha:
        verdict = STATE_CERTIFIED
    elif lower > alpha:
        verdict = STATE_VIOLATED
    else:
        verdict = STATE_UNRESOLVED
    return verdict, lower, upper


def two_sided_ci95(k: int, n: int) -> Tuple[float, float]:
    """Return a descriptive two-sided 95% Clopper--Pearson interval."""
    k = _as_int(k, "k")
    n = _as_int(n, "n")
    if not (0 <= k <= n and n >= 1):
        raise ValueError(
            f"counts must satisfy 0 <= k <= n and n >= 1; got k={k}, n={n}"
        )
    lower = float(beta.ppf(0.025, k, n - k + 1)) if k > 0 else 0.0
    upper = float(beta.ppf(0.975, k + 1, n - k)) if k < n else 1.0
    return lower, upper
