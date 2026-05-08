#!/usr/bin/env python3
"""Run a small example SAVER edit stream with an EasyEdit-backed editor."""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

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
from saver.types import ProxyParams, SaverConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=PROJECT_ROOT / "configs" / "examples" / "rome_example.json",
        help="Path to a JSON config file.",
    )
    return parser.parse_args()


def _jsonable(summary) -> dict:
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
        "snapshots": [
            {
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
                "martingales": snapshot.martingales,
                "peak_martingales": snapshot.peak_martingales,
                "oracle_risks": snapshot.oracle_risks,
                "estimated_risks": snapshot.estimated_risks,
                "oracle_gate_pass": snapshot.oracle_gate_pass,
                "martingale_pass": snapshot.martingale_pass,
                "feasible_betas": snapshot.feasible_betas,
            }
            for snapshot in summary.snapshots
        ],
    }


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


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        config = apply_env_editor_overrides(json.load(handle))

    proxy_weights = config["proxy_weights"]
    saver_config = SaverConfig(
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

    edits = load_counterfact_like_jsonl(PROJECT_ROOT / config["dataset_path"])
    limit = int(config.get("limit", len(edits)))
    edits = edits[:limit]
    ppl_text_path = None
    if config.get("ppl_text_path"):
        ppl_text_path = PROJECT_ROOT / config["ppl_text_path"]
    ppl_texts = load_ppl_texts(ppl_text_path) if ppl_text_path is not None else None

    try:
        adapter = EasyEditAdapter(
            method=config["editor"]["method"],
            hparams_path=PROJECT_ROOT / config["editor"]["hparams_path"],
            overrides=_editor_overrides(config["editor"], mode="saver"),
        )
    except ImportError as exc:
        raise SystemExit(
            f"{exc}\n"
            "Run `python3 scripts/check_environment.py --config configs/examples/rome_example.json` "
            "to see which dependency is still missing."
        ) from exc
    monitor = SaverMonitor(saver_config)
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
    if ppl_texts is not None:
        ppl_summary = dict(
            causal_lm_perplexity(
                model=adapter.model,
                tokenizer=adapter.tokenizer,
                texts=ppl_texts,
                max_length=int(config.get("max_prompt_tokens", 256)),
            )
        )

    payload = _jsonable(summary)
    payload["audit"] = audit
    payload["ppl"] = ppl_summary
    payload["editor_runtime"] = {
        "mode": "saver",
        "resolved_overrides": _editor_overrides(config["editor"], mode="saver"),
        "comparison_caveat": _comparison_caveat(config["editor"]),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
