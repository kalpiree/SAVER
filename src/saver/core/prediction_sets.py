"""Prediction-set utilities for SAVER."""

from __future__ import annotations

from typing import Dict, Iterable, List, Set, Tuple


def normalize_distribution(probabilities: Dict[str, float]) -> Dict[str, float]:
    """Normalize a token distribution and guard against degenerate input."""

    total = sum(max(value, 0.0) for value in probabilities.values())
    if total <= 0.0:
        raise ValueError("Probability mass must be positive.")
    return {token: max(value, 0.0) / total for token, value in probabilities.items()}


def sorted_mass(probabilities: Dict[str, float]) -> List[Tuple[str, float]]:
    """Return tokens ordered from highest to lowest probability."""

    normalized = normalize_distribution(probabilities)
    return sorted(normalized.items(), key=lambda item: item[1], reverse=True)


def top_mass_prediction_set(probabilities: Dict[str, float], beta: float) -> Set[str]:
    """Construct the smallest top-mass prediction set with cumulative mass >= beta."""

    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must lie in [0, 1].")

    if beta == 0.0:
        return set()

    cumulative = 0.0
    chosen: Set[str] = set()
    for token, score in sorted_mass(probabilities):
        chosen.add(token)
        cumulative += score
        if cumulative >= beta:
            break
    return chosen
