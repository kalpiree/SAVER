"""High-level SAVER monitor operating on a discrete beta grid."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Sequence

from saver.core.martingale import optimize_lambda, update_martingale
from saver.core.sampling import (
    importance_sampled_risk,
    prior_risk_estimate,
    sampling_probability,
)
from saver.types import SaverConfig, StepPlan, StepSnapshot


@dataclass
class _BetaState:
    martingale: float = 1.0
    peak_martingale: float = 1.0
    risk_history: list[float] = field(default_factory=list)
    lambda_history: list[float] = field(default_factory=list)


class SaverMonitor:
    """Implements the sequential decision logic for a fixed beta grid."""

    def __init__(self, config: SaverConfig) -> None:
        if not config.beta_grid:
            raise ValueError("beta_grid must not be empty.")
        if config.rejection_policy not in {"stop", "continue"}:
            raise ValueError("rejection_policy must be 'stop' or 'continue'.")
        if config.sampling_policy not in {"risk_adaptive", "fixed"}:
            raise ValueError("sampling_policy must be 'risk_adaptive' or 'fixed'.")
        if config.sampling_policy == "fixed":
            if config.fixed_q is None or not 0.0 < config.fixed_q <= 1.0:
                raise ValueError("fixed_q must lie in (0, 1] when sampling_policy='fixed'.")
        if config.boundary_policy not in {"adaptive", "fixed"}:
            raise ValueError("boundary_policy must be 'adaptive' or 'fixed'.")
        if config.boundary_policy == "fixed":
            if config.fixed_beta is None:
                raise ValueError("fixed_beta is required when boundary_policy='fixed'.")
            if float(config.fixed_beta) not in [float(beta) for beta in config.beta_grid]:
                raise ValueError("fixed_beta must belong to beta_grid.")
        self.config = config
        self.beta_grid = sorted(config.beta_grid)
        self.step_count = 0
        self.last_full_eval_step = 0
        self.last_chosen_beta: Optional[float] = None
        self.states = {beta: _BetaState() for beta in self.beta_grid}

    @property
    def threshold(self) -> float:
        return 1.0 / self.config.alpha

    @property
    def boundary_beta(self) -> Optional[float]:
        return self.last_chosen_beta

    @property
    def beta_max(self) -> float:
        return self.beta_grid[-1]

    def plan_step(
        self,
        structural_tension: float,
        rng: Optional[random.Random] = None,
        force_sample: Optional[bool] = None,
    ) -> StepPlan:
        """Plan sampling for one candidate edit before any oracle evaluation."""

        if rng is None:
            rng = random.Random()

        self.step_count += 1
        drift = self.step_count - self.last_full_eval_step
        r_hat = prior_risk_estimate(structural_tension, drift, self.config.proxy_params)
        if self.config.sampling_policy == "risk_adaptive":
            q_t = sampling_probability(r_hat, self.config.theta, self.config.q_min)
        else:
            assert self.config.fixed_q is not None
            q_t = self.config.fixed_q

        sampled = force_sample if force_sample is not None else (rng.random() < q_t)
        beta_floor = self.last_chosen_beta if self.config.monotone_beta_search else None
        return StepPlan(
            step=self.step_count,
            structural_tension=structural_tension,
            drift=drift,
            r_hat=r_hat,
            q_t=q_t,
            sampled=sampled,
            beta_floor=beta_floor,
        )

    def evaluate_candidate(
        self,
        plan: StepPlan,
        oracle_risks: Optional[Mapping[float, float]] = None,
    ) -> StepSnapshot:
        """Preview one candidate edit without mutating committed monitor state."""

        if plan.sampled and oracle_risks is None:
            raise ValueError("oracle_risks are required when a step is sampled.")

        martingales: Dict[float, float] = {}
        peak_martingales: Dict[float, float] = {}
        estimated_risks: Dict[float, float] = {}
        lambdas: Dict[float, float] = {}
        oracle_gate_pass: Dict[float, bool] = {}
        martingale_pass: Dict[float, bool] = {}
        recorded_oracle_risks: Dict[float, float] = {}

        for beta in self.beta_grid:
            if plan.sampled:
                assert oracle_risks is not None
                if beta not in oracle_risks:
                    raise KeyError(f"Missing oracle risk for beta={beta}.")
                observed = oracle_risks[beta]
                recorded_oracle_risks[beta] = observed
            else:
                observed = plan.r_hat

            estimated = importance_sampled_risk(
                observed_risk=observed,
                r_hat=plan.r_hat,
                sampled=plan.sampled,
                q_t=plan.q_t,
                clip_min=self.config.clip_estimated_risk_min,
                use_control_variate_proxy=self.config.use_control_variate_proxy,
            )

            state = self.states[beta]
            lambda_t = optimize_lambda(
                history=state.risk_history,
                theta=self.config.theta,
                q_min=self.config.q_min,
            )
            next_martingale = update_martingale(
                current_value=state.martingale,
                lambda_t=lambda_t,
                risk_t=estimated,
                theta=self.config.theta,
            )

            martingales[beta] = next_martingale
            peak_martingales[beta] = max(state.peak_martingale, next_martingale)
            estimated_risks[beta] = estimated
            lambdas[beta] = lambda_t
            oracle_gate_pass[beta] = (
                (not plan.sampled)
                or (not self.config.hard_gate_sampled_risk)
                or (recorded_oracle_risks[beta] <= self.config.theta)
            )
            martingale_pass[beta] = martingales[beta] < self.threshold

        feasible = [
            beta
            for beta in self.beta_grid
            if (plan.beta_floor is None or beta >= plan.beta_floor)
            and oracle_gate_pass[beta]
            and martingale_pass[beta]
        ]
        if self.config.boundary_policy == "fixed":
            assert self.config.fixed_beta is not None
            chosen_beta = self.config.fixed_beta if self.config.fixed_beta in feasible else None
        else:
            chosen_beta = feasible[0] if feasible else None
        boundary_beta = chosen_beta if chosen_beta is not None else self.last_chosen_beta

        return StepSnapshot(
            step=plan.step,
            structural_tension=plan.structural_tension,
            drift=plan.drift,
            r_hat=plan.r_hat,
            q_t=plan.q_t,
            sampled=plan.sampled,
            chosen_beta=chosen_beta,
            stop_triggered=False,
            candidate_rejected=(chosen_beta is None),
            boundary_beta=boundary_beta,
            beta_floor=plan.beta_floor,
            martingales=martingales,
            peak_martingales=peak_martingales,
            oracle_risks=recorded_oracle_risks,
            estimated_risks=estimated_risks,
            lambdas=lambdas,
            oracle_gate_pass=oracle_gate_pass,
            martingale_pass=martingale_pass,
            feasible_betas=feasible,
        )

    def observe_attempt(self, snapshot: StepSnapshot) -> None:
        """Update attempt-level bookkeeping that is independent of acceptance."""

        if snapshot.sampled:
            self.last_full_eval_step = snapshot.step

    def accept(self, snapshot: StepSnapshot) -> None:
        """Commit the accepted edit into the irreversible global boundary state."""

        if snapshot.candidate_rejected or snapshot.chosen_beta is None:
            raise ValueError("Cannot accept a rejected snapshot.")
        for beta in self.beta_grid:
            state = self.states[beta]
            state.martingale = snapshot.martingales[beta]
            state.peak_martingale = snapshot.peak_martingales[beta]
            state.risk_history.append(snapshot.estimated_risks[beta])
            state.lambda_history.append(snapshot.lambdas[beta])
        self.last_chosen_beta = snapshot.chosen_beta

    def boundary_saturated(self, snapshot: StepSnapshot) -> bool:
        """Return whether the committed boundary plus martingale evidence imply exhaustion."""

        if self.config.boundary_policy == "fixed":
            assert self.config.fixed_beta is not None
            return (
                self.config.stop_on_boundary_saturation
                and self.states[self.config.fixed_beta].peak_martingale >= self.threshold
            )
        return (
            self.config.stop_on_boundary_saturation
            and self.boundary_beta is not None
            and self.boundary_beta >= self.beta_max
            and self.states[self.beta_max].peak_martingale >= self.threshold
        )

    def step(
        self,
        structural_tension: float,
        oracle_risks: Mapping[float, float],
        rng: Optional[random.Random] = None,
        force_sample: Optional[bool] = None,
    ) -> StepSnapshot:
        """Convenience wrapper that previews and commits a feasible synthetic step."""

        plan = self.plan_step(
            structural_tension=structural_tension,
            rng=rng,
            force_sample=force_sample,
        )
        snapshot = self.evaluate_candidate(plan=plan, oracle_risks=oracle_risks)
        self.observe_attempt(snapshot)
        if not snapshot.candidate_rejected:
            snapshot.candidate_committed = True
            self.accept(snapshot)
            if self.boundary_saturated(snapshot):
                snapshot.stop_triggered = True
                snapshot.stop_reason = "boundary_evidence_exhausted"
        elif self.boundary_saturated(snapshot):
            snapshot.stop_triggered = True
            snapshot.stop_reason = "boundary_evidence_exhausted"
        return snapshot
