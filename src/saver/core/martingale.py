"""Anytime-valid martingale bookkeeping for SAVER."""

from __future__ import annotations

import math
from typing import Iterable, Sequence


def lambda_max(theta: float, q_min: float) -> float:
    """Upper bound required for a non-negative betting factor."""

    upper = 1.0 / ((1.0 / q_min) - 1.0 + theta)
    # Stay strictly inside the admissible interval so the next betting factor
    # cannot land exactly on zero due to endpoint selection.
    return math.nextafter(upper, 0.0)


def optimize_lambda(
    history: Sequence[float],
    theta: float,
    q_min: float,
    grid_size: int = 64,
) -> float:
    """Simple grid-search approximation of the predictable lambda update."""

    max_lambda = lambda_max(theta, q_min)
    if not history:
        return 0.0

    best_lambda = 0.0
    best_value = float("-inf")
    for index in range(grid_size + 1):
        candidate = max_lambda * index / grid_size
        value = 0.0
        valid = True
        for risk in history:
            factor = 1.0 + candidate * (risk - theta)
            if factor <= 0.0:
                valid = False
                break
            value += math.log(factor)
        if valid and value > best_value:
            best_value = value
            best_lambda = candidate
    return best_lambda


def update_martingale(current_value: float, lambda_t: float, risk_t: float, theta: float) -> float:
    """One-step multiplicative martingale update."""

    factor = 1.0 + lambda_t * (risk_t - theta)
    if factor <= 0.0:
        raise ValueError("Betting factor became non-positive; check q_min and risk bounds.")
    return current_value * factor
