#!/usr/bin/env python3
"""Run sequential editor-backed editing and audit ESR/PSR/NSR at checkpoints."""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
from typing import Iterable, List, Sequence

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from saver.core.monitor import SaverMonitor
from saver.core.proxy import StructuralTensionScorer
from saver.core.text_embedding import hashed_text_embedding
from saver.data.counterfact import load_counterfact_like_jsonl
from saver.editors.easyedit import EasyEditAdapter
from saver.eval.metrics import causal_lm_perplexity, load_ppl_texts, score_counterfact_metrics
from saver.eval.counterfact import CounterFactProbeGenerator
from saver.eval.first_token import FirstTokenCausalLMEvaluator
from saver.types import EditRequest, ProxyParams, SaverConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=PROJECT_ROOT / "configs" / "main" / "alphaedit_counterfact_qwen25_500.json",
        help="Path to a JSON config file.",
    )
    parser.add_argument(
        "--mode",
        choices=("saver", "unconstrained"),
        required=True,
        help="Whether to run the SAVER-gated or unconstrained sequential path.",
    )
    parser.add_argument(
        "--every",
        type=int,
        default=5,
        help="Audit every N attempted edits.",
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default="",
        help="Optional comma-separated attempted-step checkpoints, e.g. 5,10,20.",
    )
    parser.add_argument(
        "--ppl-text-path",
        type=pathlib.Path,
        default=None,
        help="Optional .txt or .jsonl corpus for final-model perplexity at checkpoints.",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help="Optional JSON path to save checkpoint results.",
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
    )


def _parse_checkpoints(raw: str, every: int, limit: int) -> List[int]:
    requested = set()
    if every > 0:
        requested.update(range(every, limit + 1, every))
    if raw.strip():
        requested.update(int(token.strip()) for token in raw.split(",") if token.strip())
    requested.add(limit)
    return sorted(step for step in requested if 1 <= step <= limit)


def _audit_record(
    *,
    step: int,
    mode: str,
    committed_edits: Sequence[EditRequest],
    attempted_steps: int,
    committed_steps: int,
    rejected_steps: int,
    final_boundary_beta: float | None,
    adapter: EasyEditAdapter,
    max_prompt_tokens: int,
    ppl_texts: Sequence[str] | None,
) -> dict:
    audit = score_counterfact_metrics(
        model=adapter.model,
        tokenizer=adapter.tokenizer,
        edits=committed_edits,
        max_prompt_tokens=max_prompt_tokens,
    )
    ppl_summary = {
        "ppl": None,
        "ppl_text_count": 0,
        "ppl_token_count": 0,
    }
    if ppl_texts is not None:
        ppl_summary = dict(
            causal_lm_perplexity(
                model=adapter.model,
                tokenizer=adapter.tokenizer,
                texts=ppl_texts,
                max_length=max_prompt_tokens,
            )
        )

    return {
        "step": step,
        "mode": mode,
        "attempted_steps": attempted_steps,
        "committed_steps": committed_steps,
        "rejected_steps": rejected_steps,
        "acceptance_rate": (committed_steps / attempted_steps) if attempted_steps else 0.0,
        "final_boundary_beta": final_boundary_beta,
        "esr": audit["esr"],
        "psr": audit["psr"],
        "ptsr": audit["ptsr"],
        "nsr": audit["nsr"],
        "rewrite_prompt_count": audit["rewrite_prompt_count"],
        "paraphrase_prompt_count": audit["paraphrase_prompt_count"],
        "portability_prompt_count": audit["portability_prompt_count"],
        "locality_prompt_count": audit["locality_prompt_count"],
        "ppl": ppl_summary["ppl"],
        "ppl_text_count": ppl_summary["ppl_text_count"],
        "ppl_token_count": ppl_summary["ppl_token_count"],
    }


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    edits = load_counterfact_like_jsonl(PROJECT_ROOT / config["dataset_path"])
    limit = int(config.get("limit", len(edits)))
    edits = edits[:limit]
    checkpoints = _parse_checkpoints(args.checkpoints, args.every, len(edits))
    ppl_text_path = args.ppl_text_path
    if ppl_text_path is None and config.get("ppl_text_path"):
        ppl_text_path = PROJECT_ROOT / config["ppl_text_path"]
    ppl_texts = load_ppl_texts(ppl_text_path) if ppl_text_path is not None else None

    try:
        adapter = EasyEditAdapter(
            method=config["editor"]["method"],
            hparams_path=PROJECT_ROOT / config["editor"]["hparams_path"],
            overrides=_editor_overrides(config["editor"], mode=args.mode),
        )
    except ImportError as exc:
        raise SystemExit(
            f"{exc}\n"
            "Run `python3 scripts/check_environment.py --config configs/examples/rome_example.json` "
            "to see which dependency is still missing."
        ) from exc

    probe_generator = CounterFactProbeGenerator()
    evaluator = FirstTokenCausalLMEvaluator(
        max_prompt_tokens=int(config.get("max_prompt_tokens", 256))
    )
    max_prompt_tokens = int(config.get("max_prompt_tokens", 256))

    attempted_steps = 0
    committed_steps = 0
    rejected_steps = 0
    records: List[dict] = []
    committed_edits: List[EditRequest] = []

    if args.mode == "saver":
        monitor = SaverMonitor(_build_saver_config(config))
        tension_scorer = StructuralTensionScorer(history_k=int(config["history_k"]))
        embedding_dim = int(config["embedding_dim"])
        committed_embeddings: List[List[float]] = []
        rng = random.Random(int(config["seed"]))

        for edit in edits:
            attempted_steps += 1
            probe_bundle = probe_generator.build(edit)
            proposal = adapter.propose(probe_bundle)
            current_embedding = list(hashed_text_embedding(probe_bundle.edit_prompt, embedding_dim))
            structural_tension = tension_scorer.score(current_embedding, committed_embeddings)
            plan = monitor.plan_step(structural_tension=structural_tension, rng=rng)

            oracle_risks = None
            if plan.sampled:
                evaluation = evaluator.evaluate(
                    proposal=proposal,
                    probe_bundle=probe_bundle,
                    beta_grid=monitor.config.beta_grid,
                    locality_weight=float(config["locality_weight"]),
                )
                oracle_risks = evaluation.joint_risk

            snapshot = monitor.evaluate_candidate(plan=plan, oracle_risks=oracle_risks)
            monitor.observe_attempt(snapshot)

            if snapshot.candidate_rejected:
                adapter.rollback(proposal)
                rejected_steps += 1
            else:
                adapter.commit(proposal)
                snapshot.candidate_committed = True
                monitor.accept(snapshot)
                committed_steps += 1
                committed_edits.append(edit)
                committed_embeddings.append(current_embedding)

            if attempted_steps in checkpoints:
                records.append(
                    _audit_record(
                        step=attempted_steps,
                        mode=args.mode,
                        committed_edits=committed_edits,
                        attempted_steps=attempted_steps,
                        committed_steps=committed_steps,
                        rejected_steps=rejected_steps,
                        final_boundary_beta=monitor.boundary_beta,
                        adapter=adapter,
                        max_prompt_tokens=max_prompt_tokens,
                        ppl_texts=ppl_texts,
                    )
                )
    else:
        for edit in edits:
            attempted_steps += 1
            proposal = adapter.propose(probe_generator.build(edit))
            adapter.commit(proposal)
            committed_steps += 1
            committed_edits.append(edit)

            if attempted_steps in checkpoints:
                records.append(
                    _audit_record(
                        step=attempted_steps,
                        mode=args.mode,
                        committed_edits=committed_edits,
                        attempted_steps=attempted_steps,
                        committed_steps=committed_steps,
                        rejected_steps=rejected_steps,
                        final_boundary_beta=None,
                        adapter=adapter,
                        max_prompt_tokens=max_prompt_tokens,
                        ppl_texts=ppl_texts,
                    )
                )

    result = {
        "mode": args.mode,
        "config_path": str(args.config),
        "dataset_path": config["dataset_path"],
        "theta": float(config["theta"]),
        "beta_grid": [float(value) for value in config["beta_grid"]],
        "checkpoints": checkpoints,
        "records": records,
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

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
