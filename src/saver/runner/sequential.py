"""Generic sequential runner for the SAVER workflow."""

from __future__ import annotations

import random
from typing import Callable, List, Sequence

from saver.core.monitor import SaverMonitor
from saver.core.proxy import StructuralTensionScorer
from saver.editors.base import BaseEditorAdapter
from saver.eval.base import BaseProbeGenerator, BaseRiskEvaluator
from saver.types import EditRequest, ExperimentSummary, StepSnapshot


EmbeddingFn = Callable[[str], Sequence[float]]


class SequentialEditRunner:
    """Orchestrates propose/evaluate/commit over an edit stream."""

    def __init__(
        self,
        monitor: SaverMonitor,
        probe_generator: BaseProbeGenerator,
        risk_evaluator: BaseRiskEvaluator,
        editor: BaseEditorAdapter,
        embedding_fn: EmbeddingFn,
        rng: random.Random | None = None,
    ) -> None:
        self.monitor = monitor
        self.probe_generator = probe_generator
        self.risk_evaluator = risk_evaluator
        self.editor = editor
        self.embedding_fn = embedding_fn
        self.tension_scorer = StructuralTensionScorer(history_k=monitor.config.history_k)
        self.rng = rng or random.Random(0)

    def run(
        self,
        edits: Sequence[EditRequest],
        locality_weight: float,
    ) -> ExperimentSummary:
        committed_embeddings: List[List[float]] = []
        snapshots: List[StepSnapshot] = []
        chosen_betas: List[float | None] = []
        rejected_steps = 0
        stop_reason: str | None = None

        for edit_request in edits:
            probe_bundle = self.probe_generator.build(edit_request)
            proposal = self.editor.propose(probe_bundle)
            current_embedding = list(self.embedding_fn(probe_bundle.edit_prompt))
            structural_tension = self.tension_scorer.score(current_embedding, committed_embeddings)
            plan = self.monitor.plan_step(
                structural_tension=structural_tension,
                rng=self.rng,
            )

            oracle_risks = None
            if plan.sampled:
                evaluation = self.risk_evaluator.evaluate(
                    proposal=proposal,
                    probe_bundle=probe_bundle,
                    beta_grid=self.monitor.config.beta_grid,
                    locality_weight=locality_weight,
                )
                oracle_risks = evaluation.joint_risk

            snapshot = self.monitor.evaluate_candidate(
                plan=plan,
                oracle_risks=oracle_risks,
            )
            chosen_betas.append(snapshot.chosen_beta)
            self.monitor.observe_attempt(snapshot)

            if snapshot.candidate_rejected:
                self.editor.rollback(proposal)
                rejected_steps += 1

                if self.monitor.boundary_saturated(snapshot):
                    snapshot.stop_triggered = True
                    snapshot.stop_reason = "boundary_evidence_exhausted"
                    stop_reason = snapshot.stop_reason
                if self.monitor.config.rejection_policy == "stop":
                    snapshot.stop_triggered = True
                    snapshot.stop_reason = "rejected_edit"
                    stop_reason = snapshot.stop_reason

                snapshots.append(snapshot)
                if snapshot.stop_triggered:
                    break
                continue

            self.editor.commit(proposal)
            snapshot.candidate_committed = True
            self.monitor.accept(snapshot)
            if self.monitor.boundary_saturated(snapshot):
                snapshot.stop_triggered = True
                snapshot.stop_reason = "boundary_evidence_exhausted"
                stop_reason = snapshot.stop_reason
            committed_embeddings.append(current_embedding)
            snapshots.append(snapshot)
            if snapshot.stop_triggered:
                break

        total_samples = sum(1 for snapshot in snapshots if snapshot.sampled)
        attempted_steps = len(snapshots)
        committed_steps = len(committed_embeddings)
        stopped_at = snapshots[-1].step if snapshots and snapshots[-1].stop_triggered else None
        acceptance_rate = (committed_steps / attempted_steps) if attempted_steps else 0.0

        return ExperimentSummary(
            attempted_steps=attempted_steps,
            committed_steps=committed_steps,
            rejected_steps=rejected_steps,
            stopped_at=stopped_at,
            stop_reason=stop_reason,
            total_samples=total_samples,
            acceptance_rate=acceptance_rate,
            final_boundary_beta=self.monitor.boundary_beta,
            chosen_betas=chosen_betas,
            snapshots=snapshots,
        )
