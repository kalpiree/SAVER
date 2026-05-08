#!/usr/bin/env python3
"""Run sequential editing and save hidden-state snapshots for representation-drift plots."""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
from typing import List, Sequence

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from saver.core.monitor import SaverMonitor
from saver.core.proxy import StructuralTensionScorer
from saver.core.text_embedding import hashed_text_embedding
from saver.data.counterfact import load_counterfact_like_jsonl
from saver.editors.easyedit import EasyEditAdapter
from saver.eval.counterfact import CounterFactProbeGenerator
from saver.eval.first_token import FirstTokenCausalLMEvaluator
from saver.types import EditRequest, ProxyParams, SaverConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        required=True,
        help="Path to a JSON config file.",
    )
    parser.add_argument(
        "--mode",
        choices=("saver", "unconstrained"),
        required=True,
        help="Whether to run the SAVER-gated or unconstrained sequential path.",
    )
    parser.add_argument(
        "--output-prefix",
        type=pathlib.Path,
        required=True,
        help="Prefix for the saved .json and .npz outputs.",
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default="0,100,250,500",
        help="Comma-separated attempted-step checkpoints, including 0 if desired.",
    )
    parser.add_argument(
        "--probe-count",
        type=int,
        default=256,
        help="Number of untouched held-out probe prompts to encode.",
    )
    parser.add_argument(
        "--probe-offset",
        type=int,
        default=None,
        help="Start index for held-out probe prompts. Defaults to config['limit'].",
    )
    parser.add_argument(
        "--probe-batch-size",
        type=int,
        default=8,
        help="Probe encoding batch size.",
    )
    parser.add_argument(
        "--layer-index",
        type=int,
        default=-1,
        help="Hidden-state layer index to encode. -1 means the final layer.",
    )
    parser.add_argument(
        "--pooling",
        choices=("mean", "last"),
        default="mean",
        help="How to reduce token hidden states into one vector per probe.",
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
        "--model-path",
        type=str,
        default=None,
        help="Optional override for editor.model_name_override.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Optional override for editor.tokenizer_name_override.",
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


def _parse_checkpoints(raw: str, limit: int) -> List[int]:
    checkpoints = sorted(
        {
            int(token.strip())
            for token in raw.split(",")
            if token.strip()
        }
    )
    return [step for step in checkpoints if 0 <= step <= limit]


def _heldout_probe_prompts(
    records: Sequence[EditRequest],
    *,
    offset: int,
    count: int,
) -> List[str]:
    prompts: List[str] = []
    seen = set()
    for record in records[offset:]:
        prompt = record.metadata.get("rewrite_prompt")
        if not isinstance(prompt, str) or not prompt or prompt in seen:
            continue
        prompts.append(prompt)
        seen.add(prompt)
        if len(prompts) >= count:
            break
    if len(prompts) < count:
        raise ValueError(
            f"Only found {len(prompts)} held-out probe prompts after offset {offset}; "
            f"need {count}."
        )
    return prompts


def _model_device(model, torch_module):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch_module.device("cpu")


def _encode_prompts(
    *,
    model,
    tokenizer,
    prompts: Sequence[str],
    max_prompt_tokens: int,
    probe_batch_size: int,
    layer_index: int,
    pooling: str,
    torch_module,
) -> np.ndarray:
    device = _model_device(model, torch_module)
    model.eval()
    batches: List[np.ndarray] = []
    with torch_module.no_grad():
        for start in range(0, len(prompts), probe_batch_size):
            batch_prompts = list(prompts[start : start + probe_batch_size])
            encoded = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_prompt_tokens,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = model(**encoded, output_hidden_states=True)
            hidden = outputs.hidden_states[layer_index]
            mask = encoded["attention_mask"]

            if pooling == "mean":
                weights = mask.unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
            else:
                last_indices = mask.sum(dim=1) - 1
                pooled = hidden[torch_module.arange(hidden.shape[0], device=device), last_indices]

            batches.append(pooled.detach().cpu().to(torch_module.float32).numpy())
    return np.concatenate(batches, axis=0)


def _drift_stats(reference: np.ndarray, current: np.ndarray) -> dict:
    ref = reference.astype(np.float64, copy=False)
    cur = current.astype(np.float64, copy=False)
    delta = cur - ref
    mean_l2 = float(np.linalg.norm(delta, axis=1).mean())
    ref_centroid = ref.mean(axis=0)
    cur_centroid = cur.mean(axis=0)
    centroid_l2 = float(np.linalg.norm(cur_centroid - ref_centroid))

    ref_norm = np.linalg.norm(ref, axis=1)
    cur_norm = np.linalg.norm(cur, axis=1)
    denom = np.clip(ref_norm * cur_norm, 1e-12, None)
    cosine = np.sum(ref * cur, axis=1) / denom
    mean_cosine = float(cosine.mean())
    return {
        "mean_l2_from_step0": mean_l2,
        "centroid_l2_from_step0": centroid_l2,
        "mean_cosine_to_step0": mean_cosine,
    }


def _checkpoint_record(
    *,
    step: int,
    attempted_steps: int,
    committed_steps: int,
    rejected_steps: int,
    final_boundary_beta: float | None,
    reference_repr: np.ndarray,
    current_repr: np.ndarray,
) -> dict:
    return {
        "step": step,
        "attempted_steps": attempted_steps,
        "committed_steps": committed_steps,
        "rejected_steps": rejected_steps,
        "acceptance_rate": (committed_steps / attempted_steps) if attempted_steps else None,
        "final_boundary_beta": final_boundary_beta,
        **_drift_stats(reference_repr, current_repr),
    }


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    if args.theta is not None:
        config["theta"] = float(args.theta)
    if args.alpha is not None:
        config["alpha"] = float(args.alpha)
    if args.seed is not None:
        config["seed"] = int(args.seed)
    if args.model_path is not None:
        config["editor"]["model_name_override"] = args.model_path
    if args.tokenizer_path is not None:
        config["editor"]["tokenizer_name_override"] = args.tokenizer_path

    all_records = load_counterfact_like_jsonl(PROJECT_ROOT / config["dataset_path"])
    limit = int(config.get("limit", len(all_records)))
    edits = all_records[:limit]
    checkpoints = _parse_checkpoints(args.checkpoints, len(edits))
    probe_offset = args.probe_offset if args.probe_offset is not None else len(edits)
    probe_prompts = _heldout_probe_prompts(
        all_records,
        offset=probe_offset,
        count=args.probe_count,
    )
    max_prompt_tokens = int(config.get("max_prompt_tokens", 256))

    adapter = EasyEditAdapter(
        method=config["editor"]["method"],
        hparams_path=PROJECT_ROOT / config["editor"]["hparams_path"],
        overrides=_editor_overrides(config["editor"], mode=args.mode),
    )
    probe_generator = CounterFactProbeGenerator()
    evaluator = FirstTokenCausalLMEvaluator(max_prompt_tokens=max_prompt_tokens)

    arrays = {}
    records = []

    step0_repr = _encode_prompts(
        model=adapter.model,
        tokenizer=adapter.tokenizer,
        prompts=probe_prompts,
        max_prompt_tokens=max_prompt_tokens,
        probe_batch_size=args.probe_batch_size,
        layer_index=args.layer_index,
        pooling=args.pooling,
        torch_module=adapter.torch,
    )
    arrays["step_0000"] = step0_repr.astype(np.float32)
    records.append(
        _checkpoint_record(
            step=0,
            attempted_steps=0,
            committed_steps=0,
            rejected_steps=0,
            final_boundary_beta=None,
            reference_repr=step0_repr,
            current_repr=step0_repr,
        )
    )

    attempted_steps = 0
    committed_steps = 0
    rejected_steps = 0

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
                committed_embeddings.append(current_embedding)

            if attempted_steps in checkpoints:
                current_repr = _encode_prompts(
                    model=adapter.model,
                    tokenizer=adapter.tokenizer,
                    prompts=probe_prompts,
                    max_prompt_tokens=max_prompt_tokens,
                    probe_batch_size=args.probe_batch_size,
                    layer_index=args.layer_index,
                    pooling=args.pooling,
                    torch_module=adapter.torch,
                )
                arrays[f"step_{attempted_steps:04d}"] = current_repr.astype(np.float32)
                records.append(
                    _checkpoint_record(
                        step=attempted_steps,
                        attempted_steps=attempted_steps,
                        committed_steps=committed_steps,
                        rejected_steps=rejected_steps,
                        final_boundary_beta=monitor.boundary_beta,
                        reference_repr=step0_repr,
                        current_repr=current_repr,
                    )
                )
    else:
        for edit in edits:
            attempted_steps += 1
            proposal = adapter.propose(probe_generator.build(edit))
            adapter.commit(proposal)
            committed_steps += 1

            if attempted_steps in checkpoints:
                current_repr = _encode_prompts(
                    model=adapter.model,
                    tokenizer=adapter.tokenizer,
                    prompts=probe_prompts,
                    max_prompt_tokens=max_prompt_tokens,
                    probe_batch_size=args.probe_batch_size,
                    layer_index=args.layer_index,
                    pooling=args.pooling,
                    torch_module=adapter.torch,
                )
                arrays[f"step_{attempted_steps:04d}"] = current_repr.astype(np.float32)
                records.append(
                    _checkpoint_record(
                        step=attempted_steps,
                        attempted_steps=attempted_steps,
                        committed_steps=committed_steps,
                        rejected_steps=rejected_steps,
                        final_boundary_beta=None,
                        reference_repr=step0_repr,
                        current_repr=current_repr,
                    )
                )

    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    npz_path = prefix.with_suffix(".npz")
    json_path = prefix.with_suffix(".json")

    np.savez_compressed(npz_path, prompts=np.array(probe_prompts, dtype=object), **arrays)
    payload = {
        "mode": args.mode,
        "config_path": str(args.config),
        "dataset_path": config["dataset_path"],
        "probe_offset": probe_offset,
        "probe_count": len(probe_prompts),
        "probe_prompts_path": str(npz_path),
        "checkpoints": checkpoints,
        "layer_index": args.layer_index,
        "pooling": args.pooling,
        "records": records,
        "editor": {
            "method": config["editor"]["method"],
            "resolved_overrides": _editor_overrides(config["editor"], mode=args.mode),
        },
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
