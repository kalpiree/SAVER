#!/usr/bin/env python3
"""Run SAVER or unconstrained editor-backed sequential editing and audit the final model."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import random
import sys
import time
from typing import List, Sequence

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from saver.core.monitor import SaverMonitor
from saver.core.text_embedding import hashed_text_embedding
from saver.data.counterfact import load_counterfact_like_jsonl
from saver.editors.easyedit import EasyEditAdapter
from saver.eval.metrics import causal_lm_perplexity, load_ppl_texts, score_counterfact_metrics
from saver.eval.counterfact import CounterFactProbeGenerator
from saver.eval.first_token import FirstTokenCausalLMEvaluator
from saver.runtime_config import apply_env_editor_overrides
from saver.runner.sequential import SequentialEditRunner
from saver.types import EditRequest, ExperimentSummary, ProxyParams, SaverConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=PROJECT_ROOT / "configs" / "alphaedit_counterfact_qwen25_500.json",
        help="Path to a JSON config file.",
    )
    parser.add_argument(
        "--mode",
        choices=("saver", "unconstrained"),
        required=True,
        help="Whether to audit SAVER-gated sequential editing or the unconstrained baseline.",
    )
    parser.add_argument(
        "--ppl-text-path",
        type=pathlib.Path,
        default=None,
        help="Optional .txt or .jsonl corpus for final-model perplexity.",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help="Optional JSON path to save the audit summary.",
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=None,
        help="Optional override for config['theta'].",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Optional override for config['alpha'].",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional override for config['seed'].",
    )
    parser.add_argument(
        "--beta-grid",
        type=str,
        default=None,
        help="Optional whitespace/comma-separated override for config['beta_grid'].",
    )
    parser.add_argument(
        "--fixed-beta",
        type=float,
        default=None,
        help="Optional override for config['fixed_beta'].",
    )
    return parser.parse_args()


def _editor_overrides(editor_config: dict, *, mode: str | None = None) -> dict:
    overrides = {}
    for key, value in editor_config.items():
        if key.endswith("_override"):
            overrides[key[: -len("_override")]] = value
    if mode is not None:
        mode_overrides = editor_config.get("mode_overrides", {}).get(mode, {})
        for key, value in mode_overrides.items():
            if key.endswith("_override"):
                overrides[key[: -len("_override")]] = value
    return overrides


def _comparison_caveat(editor_config: dict) -> str | None:
    mode_overrides = editor_config.get("mode_overrides", {})
    saver_mode = _editor_overrides(editor_config, mode="saver")
    unconstrained_mode = _editor_overrides(editor_config, mode="unconstrained")
    if not mode_overrides or saver_mode == unconstrained_mode:
        return None
    return (
        "Mode-specific editor precision is active: saver and unconstrained use "
        "different EasyEdit override profiles. Treat cross-mode results as an "
        "exploratory best-achievable comparison, not a like-for-like fairness claim."
    )


def _build_saver_config(config: dict) -> SaverConfig:
    proxy_weights = config["proxy_weights"]
    return SaverConfig(
        beta_grid=[float(value) for value in config["beta_grid"]],
        theta=float(config["theta"]),
        alpha=float(config["alpha"]),
        q_min=float(config["q_min"]),
        history_k=int(config["history_k"]),
        proxy_params=ProxyParams(
            w=float(proxy_weights["w"]),
            gamma=float(proxy_weights["gamma"]),
            b=float(proxy_weights["b"]),
        ),
        hard_gate_sampled_risk=bool(config.get("hard_gate_sampled_risk", True)),
        monotone_beta_search=bool(config.get("monotone_beta_search", True)),
        clip_estimated_risk_min=float(config.get("clip_estimated_risk_min", 0.0)),
        rejection_policy=str(config.get("rejection_policy", "continue")),
        stop_on_boundary_saturation=bool(config.get("stop_on_boundary_saturation", True)),
        sampling_policy=str(config.get("sampling_policy", "risk_adaptive")),
        fixed_q=(float(config["fixed_q"]) if config.get("fixed_q") is not None else None),
        use_control_variate_proxy=bool(config.get("use_control_variate_proxy", True)),
        boundary_policy=str(config.get("boundary_policy", "adaptive")),
        fixed_beta=(
            float(config["fixed_beta"]) if config.get("fixed_beta") is not None else None
        ),
    )


def _summary_to_dict(summary: ExperimentSummary) -> dict:
    return {
        "attempted_steps": summary.attempted_steps,
        "committed_steps": summary.committed_steps,
        "rejected_steps": summary.rejected_steps,
        "stopped_at": summary.stopped_at,
        "stop_reason": summary.stop_reason,
        "total_samples": summary.total_samples,
        "acceptance_rate": summary.acceptance_rate,
        "final_boundary_beta": summary.final_boundary_beta,
        "chosen_betas": summary.chosen_betas,
    }


def _average(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _snapshot_to_dict(snapshot) -> dict:
    return {
        "step": snapshot.step,
        "structural_tension": snapshot.structural_tension,
        "drift": snapshot.drift,
        "r_hat": snapshot.r_hat,
        "q_t": snapshot.q_t,
        "sampled": snapshot.sampled,
        "chosen_beta": snapshot.chosen_beta,
        "stop_triggered": snapshot.stop_triggered,
        "candidate_rejected": snapshot.candidate_rejected,
        "candidate_committed": snapshot.candidate_committed,
        "stop_reason": snapshot.stop_reason,
        "boundary_beta": snapshot.boundary_beta,
        "beta_floor": snapshot.beta_floor,
        "martingales": {str(beta): value for beta, value in snapshot.martingales.items()},
        "peak_martingales": {
            str(beta): value for beta, value in snapshot.peak_martingales.items()
        },
        "oracle_risks": {str(beta): value for beta, value in snapshot.oracle_risks.items()},
        "estimated_risks": {
            str(beta): value for beta, value in snapshot.estimated_risks.items()
        },
        "lambdas": {str(beta): value for beta, value in snapshot.lambdas.items()},
        "oracle_gate_pass": {
            str(beta): value for beta, value in snapshot.oracle_gate_pass.items()
        },
        "martingale_pass": {
            str(beta): value for beta, value in snapshot.martingale_pass.items()
        },
        "feasible_betas": snapshot.feasible_betas,
    }


def _saver_metrics(summary: ExperimentSummary, *, theta: float, beta_grid: Sequence[float]) -> dict:
    accepted = [snapshot for snapshot in summary.snapshots if snapshot.candidate_committed]
    accepted_risks: List[float] = []
    chosen_betas: List[float] = []

    for snapshot in accepted:
        if snapshot.chosen_beta is None:
            continue
        beta_key = float(snapshot.chosen_beta)
        if beta_key in snapshot.oracle_risks:
            accepted_risks.append(float(snapshot.oracle_risks[beta_key]))
        chosen_betas.append(beta_key)

    per_beta_sampled_mean_risk = {}
    for beta in beta_grid:
        beta = float(beta)
        risks = [
            float(snapshot.oracle_risks[beta])
            for snapshot in accepted
            if beta in snapshot.oracle_risks
        ]
        per_beta_sampled_mean_risk[str(beta)] = _average(risks)

    violation_rate = None
    excess_risk = None
    if accepted_risks:
        violation_rate = sum(1.0 for risk in accepted_risks if risk > theta) / len(accepted_risks)
        excess_risk = sum(max(0.0, risk - theta) for risk in accepted_risks) / len(accepted_risks)

    return {
        "accepted_edits": len(accepted),
        "oracle_checks": summary.total_samples,
        "dhat": _average(accepted_risks),
        "gap": max(0.0, _average(accepted_risks) - theta) if accepted_risks else None,
        "violation_rate": violation_rate,
        "excess_risk": excess_risk,
        "mean_chosen_beta": _average(chosen_betas),
        "per_beta_sampled_mean_risk": per_beta_sampled_mean_risk,
    }


def _run_saver(
    config: dict,
    edits: Sequence[EditRequest],
    adapter: EasyEditAdapter,
) -> tuple[ExperimentSummary, List[EditRequest]]:
    monitor = SaverMonitor(_build_saver_config(config))
    runner = SequentialEditRunner(
        monitor=monitor,
        probe_generator=CounterFactProbeGenerator(),
        risk_evaluator=FirstTokenCausalLMEvaluator(
            max_prompt_tokens=int(config.get("max_prompt_tokens", 256))
        ),
        editor=adapter,
        embedding_fn=lambda text: hashed_text_embedding(text, int(config["embedding_dim"])),
        rng=random.Random(int(config["seed"])),
    )
    summary = runner.run(
        edits=edits,
        locality_weight=float(config["locality_weight"]),
    )
    committed_edits = [
        edit
        for edit, snapshot in zip(edits, summary.snapshots)
        if snapshot.candidate_committed
    ]
    return summary, committed_edits


def _run_unconstrained(
    config: dict,
    edits: Sequence[EditRequest],
    adapter: EasyEditAdapter,
) -> tuple[dict, List[EditRequest]]:
    probe_generator = CounterFactProbeGenerator()
    committed_edits: List[EditRequest] = []
    for edit in edits:
        proposal = adapter.propose(probe_generator.build(edit))
        adapter.commit(proposal)
        committed_edits.append(edit)

    return {
        "attempted_steps": len(edits),
        "committed_steps": len(committed_edits),
        "rejected_steps": 0,
        "stopped_at": None,
        "stop_reason": None,
        "total_samples": None,
        "acceptance_rate": 1.0 if edits else 0.0,
        "final_boundary_beta": None,
        "chosen_betas": [],
    }, committed_edits


def main() -> None:
    args = parse_args()
    started_at = dt.datetime.now().astimezone()
    started_monotonic = time.perf_counter()
    with args.config.open("r", encoding="utf-8") as handle:
        config = apply_env_editor_overrides(json.load(handle))
    if args.theta is not None:
        config["theta"] = float(args.theta)
    if args.alpha is not None:
        config["alpha"] = float(args.alpha)
    if args.seed is not None:
        config["seed"] = int(args.seed)
    if args.beta_grid is not None:
        tokens = args.beta_grid.replace(",", " ").split()
        config["beta_grid"] = [float(token) for token in tokens]
    if args.fixed_beta is not None:
        config["fixed_beta"] = float(args.fixed_beta)

    edits = load_counterfact_like_jsonl(PROJECT_ROOT / config["dataset_path"])
    limit = int(config.get("limit", len(edits)))
    edits = edits[:limit]
    ppl_text_path = args.ppl_text_path
    if ppl_text_path is None and config.get("ppl_text_path"):
        ppl_text_path = PROJECT_ROOT / config["ppl_text_path"]

    try:
        adapter = EasyEditAdapter(
            method=config["editor"]["method"],
            hparams_path=PROJECT_ROOT / config["editor"]["hparams_path"],
            overrides=_editor_overrides(config["editor"], mode=args.mode),
        )
    except ImportError as exc:
        raise SystemExit(
            f"{exc}\n"
            "Run `python3 scripts/check_environment.py --config configs/rome_example.json` "
            "to see which dependency is still missing."
        ) from exc

    if args.mode == "saver":
        saver_summary, committed_edits = _run_saver(
            config=config,
            edits=edits,
            adapter=adapter,
        )
        run_summary = _summary_to_dict(saver_summary)
        snapshots = [_snapshot_to_dict(snapshot) for snapshot in saver_summary.snapshots]
    else:
        run_summary, committed_edits = _run_unconstrained(
            config=config,
            edits=edits,
            adapter=adapter,
        )
        snapshots = None

    audit = score_counterfact_metrics(
        model=adapter.model,
        tokenizer=adapter.tokenizer,
        edits=committed_edits,
        max_prompt_tokens=int(config.get("max_prompt_tokens", 256)),
    )

    ppl_summary = {
        "ppl": None,
        "ppl_text_count": 0,
        "ppl_token_count": 0,
    }
    if ppl_text_path is not None:
        ppl_texts = load_ppl_texts(ppl_text_path)
        ppl_summary = dict(
            causal_lm_perplexity(
                model=adapter.model,
                tokenizer=adapter.tokenizer,
                texts=ppl_texts,
                max_length=int(config.get("max_prompt_tokens", 256)),
            )
        )

    finished_at = dt.datetime.now().astimezone()
    wall_clock_seconds = time.perf_counter() - started_monotonic

    result = {
        "mode": args.mode,
        "config_path": str(args.config),
        "dataset_path": config["dataset_path"],
        "theta": float(config["theta"]),
        "alpha": float(config["alpha"]),
        "seed": int(config["seed"]),
        "beta_grid": [float(value) for value in config["beta_grid"]],
        "timing": {
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "wall_clock_seconds": wall_clock_seconds,
        },
        "run_summary": run_summary,
        "audit": audit,
        "ppl": ppl_summary,
        "saver_runtime": {
            "sampling_policy": str(config.get("sampling_policy", "risk_adaptive")),
            "fixed_q": (
                float(config["fixed_q"]) if config.get("fixed_q") is not None else None
            ),
            "use_control_variate_proxy": bool(
                config.get("use_control_variate_proxy", True)
            ),
            "boundary_policy": str(config.get("boundary_policy", "adaptive")),
            "fixed_beta": (
                float(config["fixed_beta"]) if config.get("fixed_beta") is not None else None
            ),
        },
        "editor_runtime": {
            "mode": args.mode,
            "resolved_overrides": _editor_overrides(config["editor"], mode=args.mode),
            "comparison_caveat": _comparison_caveat(config["editor"]),
        },
        "notes": {
            "metric_unit": "first_token_exact_match",
            "mmlu": "not_computed",
            "ppl_text_path": str(ppl_text_path) if ppl_text_path is not None else None,
        },
    }
    if snapshots is not None:
        result["snapshots"] = snapshots
        result["policy_metrics"] = _saver_metrics(
            saver_summary,
            theta=float(config["theta"]),
            beta_grid=[float(value) for value in config["beta_grid"]],
        )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    stdout_result = dict(result)
    stdout_result.pop("snapshots", None)
    print(json.dumps(stdout_result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
