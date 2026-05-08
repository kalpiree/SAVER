"""Evaluation hooks for real SAVER experiments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from saver.types import EditRequest, EditorProposal, EvaluationResult, ProbeBundle


class BaseProbeGenerator(ABC):
    """Create edit, generality, and locality probes for one edit request."""

    @abstractmethod
    def build(self, edit_request: EditRequest) -> ProbeBundle:
        """Return the probe bundle for one factual edit."""


class BaseRiskEvaluator(ABC):
    """Oracle evaluator for the candidate edited model."""

    @abstractmethod
    def evaluate(
        self,
        proposal: EditorProposal,
        probe_bundle: ProbeBundle,
        beta_grid: Sequence[float],
        locality_weight: float,
    ) -> EvaluationResult:
        """Return risks across the configured beta grid."""
