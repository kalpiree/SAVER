"""Anytime-valid martingale bookkeeping for SAVER."""

from __future__ import annotations

import math
from typing import Iterable, Sequence


def lambda_max(theta: float, q_min: float) -> float:
    """Upper bound required for a non-negative betting factor."""

    upper = 1.0 / ((1.0 / q_min) - 1.0 + theta)
    return math.nextafter(upper, 0.0)


def optimize_lambda(
    history: Sequence[float],
    theta: float,
    q_min: float,
    adaptive_sampling: bool = True,
) -> float:
    max_lambda = lambda_max(theta, q_min)
    detection_cap = (
        1.0 / (2.0 * theta * (2.0 - q_min))
        if adaptive_sampling
        else max_lambda / 2.0
    )
    max_lambda = min(max_lambda, math.nextafter(detection_cap, 0.0))
    if not history:
        return 0.0
    centered = [risk - theta for risk in history]
    if sum(centered) <= 0.0:
        return 0.0
    def derivative(value: float) -> float:
        return sum(delta / (1.0 + value * delta) for delta in centered)
    if derivative(max_lambda) >= 0.0:
        return max_lambda
    low, high = 0.0, max_lambda
    for _ in range(64):
        middle = (low + high) / 2.0
        if derivative(middle) > 0.0:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def update_martingale(current_value: float, lambda_t: float, risk_t: float, theta: float) -> float:
    """One-step multiplicative martingale update."""

    factor = 1.0 + lambda_t * (risk_t - theta)
    if factor <= 0.0:
        raise ValueError("Betting factor became non-positive; check q_min and risk bounds.")
    return current_value * factor
