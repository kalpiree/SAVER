"""Risk utilities for locality and generality evaluation."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence

from saver.core.prediction_sets import top_mass_prediction_set


def miscoverage(target: str, prediction_set: Sequence[str]) -> float:
    """Indicator loss used by SAVER."""

    return 0.0 if target in prediction_set else 1.0


def average_miscoverage(
    distributions: Sequence[Mapping[str, float]],
    targets: Sequence[str],
    beta: float,
) -> float:
    """Average indicator loss across a finite probe set."""

    if len(distributions) != len(targets):
        raise ValueError("Each distribution must have a matching target.")
    if not distributions:
        return 0.0

    total = 0.0
    for distribution, target in zip(distributions, targets):
        prediction_set = top_mass_prediction_set(dict(distribution), beta)
        total += miscoverage(target, prediction_set)
    return total / len(distributions)


def joint_risk(
    locality_distributions: Sequence[Mapping[str, float]],
    locality_targets: Sequence[str],
    generality_distributions: Sequence[Mapping[str, float]],
    generality_targets: Sequence[str],
    beta: float,
    locality_weight: float,
) -> Dict[str, float]:
    """Compute locality, generality, and weighted joint risk for one beta."""

    if not 0.0 <= locality_weight <= 1.0:
        raise ValueError("locality_weight must lie in [0, 1].")

    locality = average_miscoverage(locality_distributions, locality_targets, beta)
    generality = average_miscoverage(generality_distributions, generality_targets, beta)
    combined = locality_weight * locality + (1.0 - locality_weight) * generality
    return {
        "locality": locality,
        "generality": generality,
        "joint": combined,
    }
