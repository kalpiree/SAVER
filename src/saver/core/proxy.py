"""Structural tension proxy used to prioritize expensive evaluations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence


def _check_same_length(lhs: Sequence[float], rhs: Sequence[float]) -> None:
    if len(lhs) != len(rhs):
        raise ValueError("Vectors must have the same dimensionality.")


def l2_norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def cosine_similarity(lhs: Sequence[float], rhs: Sequence[float]) -> float:
    _check_same_length(lhs, rhs)
    lhs_norm = l2_norm(lhs)
    rhs_norm = l2_norm(rhs)
    if lhs_norm == 0.0 or rhs_norm == 0.0:
        return 0.0
    numerator = sum(a * b for a, b in zip(lhs, rhs))
    return numerator / (lhs_norm * rhs_norm)


@dataclass(frozen=True)
class DiagonalWhiteningStats:
    """A lightweight diagonal approximation of covariance correction.

    This keeps the initial implementation dependency-free. When we integrate
    real sentence embeddings, we can replace this with a full whitening matrix.
    """

    mean: Sequence[float]
    scale: Sequence[float]

    def transform(self, vector: Sequence[float]) -> List[float]:
        _check_same_length(self.mean, vector)
        _check_same_length(self.scale, vector)
        output: List[float] = []
        for value, mean, scale in zip(vector, self.mean, self.scale):
            denom = scale if abs(scale) > 1e-8 else 1.0
            output.append((value - mean) / denom)
        return output


class StructuralTensionScorer:
    """Computes the crowding score against previously committed edits."""

    def __init__(self, history_k: int, whitening: DiagonalWhiteningStats | None = None) -> None:
        if history_k <= 0:
            raise ValueError("history_k must be positive.")
        self.history_k = history_k
        self.whitening = whitening

    def _prepare(self, vector: Sequence[float]) -> List[float]:
        if self.whitening is None:
            return list(vector)
        return self.whitening.transform(vector)

    def score(
        self,
        current_embedding: Sequence[float],
        previous_embeddings: Sequence[Sequence[float]],
    ) -> float:
        """Return the average similarity to the nearest committed edits."""

        if not previous_embeddings:
            return 0.0

        current = self._prepare(current_embedding)
        similarities = [
            cosine_similarity(current, self._prepare(previous))
            for previous in previous_embeddings
        ]
        similarities.sort(reverse=True)
        nearest = similarities[: self.history_k]
        return sum(nearest) / len(nearest)
