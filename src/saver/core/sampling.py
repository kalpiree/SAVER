"""Sampling and control-variate estimation rules for SAVER."""

from __future__ import annotations

import math

from saver.types import ProxyParams


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def prior_risk_estimate(structural_tension: float, drift: int, params: ProxyParams) -> float:
    """Compute r_hat_t from structural tension and time drift."""

    raw = params.w * structural_tension + params.gamma * drift + params.b
    return sigmoid(raw)


def sampling_probability(r_hat: float, theta: float, q_min: float) -> float:
    """Risk-adaptive Bernoulli sampling probability."""

    if theta <= 0.0:
        raise ValueError("theta must be positive.")
    if not 0.0 < q_min <= 1.0:
        raise ValueError("q_min must lie in (0, 1].")
    return min(1.0, max(q_min, r_hat / theta))


def importance_sampled_risk(
    observed_risk: float,
    r_hat: float,
    sampled: bool,
    q_t: float,
    clip_min: float = 0.0,
    use_control_variate_proxy: bool = True,
) -> float:
    """Control-variate estimator with optional lower clipping for stability."""

    if use_control_variate_proxy:
        if not sampled:
            return max(clip_min, r_hat)
        estimated = r_hat + (observed_risk - r_hat) / q_t
    else:
        estimated = (observed_risk / q_t) if sampled else 0.0
    return max(clip_min, estimated)
