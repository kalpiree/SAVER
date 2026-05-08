"""Shared dataclasses used across the SAVER codebase."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence


@dataclass(frozen=True)
class ProxyParams:
    """Coefficients for the structural tension risk predictor."""

    w: float
    gamma: float
    b: float


@dataclass(frozen=True)
class SaverConfig:
    """Monitor configuration shared by the synthetic and real runners."""

    beta_grid: Sequence[float]
    theta: float
    alpha: float
    q_min: float
    history_k: int
    proxy_params: ProxyParams
    hard_gate_sampled_risk: bool = True
    monotone_beta_search: bool = True
    clip_estimated_risk_min: float = 0.0
    rejection_policy: str = "continue"
    stop_on_boundary_saturation: bool = True
    sampling_policy: str = "risk_adaptive"
    fixed_q: Optional[float] = None
    use_control_variate_proxy: bool = True
    boundary_policy: str = "adaptive"
    fixed_beta: Optional[float] = None


@dataclass(frozen=True)
class EditRequest:
    """One knowledge-edit request in the sequential stream."""

    subject: str
    relation: str
    target: str
    paraphrases: Sequence[str] = ()
    locality_subjects: Sequence[str] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class StepSnapshot:
    """A compact record of one sequential monitoring step."""

    step: int
    structural_tension: float
    drift: int
    r_hat: float
    q_t: float
    sampled: bool
    chosen_beta: Optional[float]
    stop_triggered: bool
    candidate_rejected: bool = False
    candidate_committed: bool = False
    stop_reason: Optional[str] = None
    boundary_beta: Optional[float] = None
    beta_floor: Optional[float] = None
    martingales: Dict[float, float] = field(default_factory=dict)
    peak_martingales: Dict[float, float] = field(default_factory=dict)
    oracle_risks: Dict[float, float] = field(default_factory=dict)
    estimated_risks: Dict[float, float] = field(default_factory=dict)
    lambdas: Dict[float, float] = field(default_factory=dict)
    oracle_gate_pass: Dict[float, bool] = field(default_factory=dict)
    martingale_pass: Dict[float, bool] = field(default_factory=dict)
    feasible_betas: List[float] = field(default_factory=list)


@dataclass(frozen=True)
class StepPlan:
    """Sampling plan for one candidate edit before any oracle evaluation."""

    step: int
    structural_tension: float
    drift: int
    r_hat: float
    q_t: float
    sampled: bool
    beta_floor: Optional[float] = None


@dataclass(frozen=True)
class ProbeSet:
    """Probes and targets used to score locality and generality."""

    prompts: Sequence[str]
    targets: Sequence[str]


@dataclass(frozen=True)
class ProbeBundle:
    """Grouping of edit, generality, and locality probes for one edit."""

    edit_request: EditRequest
    edit_prompt: str
    generality: ProbeSet
    locality: ProbeSet


@dataclass(frozen=True)
class EditorProposal:
    """Return object for editors that separate propose from commit."""

    metadata: Mapping[str, str]
    handle: object


@dataclass
class EvaluationResult:
    """Oracle evaluation for one step across a beta grid."""

    generality_risk: Mapping[float, float]
    locality_risk: Mapping[float, float]
    joint_risk: Mapping[float, float]
    notes: Optional[Mapping[str, str]] = None


@dataclass
class ExperimentSummary:
    """Top-level results from a monitor run."""

    attempted_steps: int
    committed_steps: int
    rejected_steps: int
    stopped_at: Optional[int]
    stop_reason: Optional[str]
    total_samples: int
    acceptance_rate: float
    final_boundary_beta: Optional[float]
    chosen_betas: List[Optional[float]]
    snapshots: List[StepSnapshot]
