#!/usr/bin/env python3
"""Run the reviewer-rebuttal sequential editing stream with checkpoint audits."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import pathlib
import random
import re
import sys
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Iterable, List, Mapping, Sequence, cast

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from saver.core.monitor import SaverMonitor
from saver.core.proxy import StructuralTensionScorer
from saver.core.text_embedding import hashed_text_embedding
from saver.data.counterfact import load_counterfact_like_jsonl
from saver.editors.easyedit import EasyEditAdapter
from saver.eval.audit import (
    causal_lm_perplexity,
    first_token_exact_match,
    load_ppl_texts,
    score_counterfact_metrics,
)
from saver.eval.counterfact import CounterFactProbeGenerator
from saver.eval.first_token import FirstTokenCausalLMEvaluator
from saver.runtime_config import apply_env_editor_overrides
from saver.types import EditRequest, EditorProposal, ProbeBundle, ProbeSet, ProxyParams, SaverConfig


MODES = ("saver", "unconstrained", "random_reject", "probe_gate", "kl_gate")
PROBE_DESIGNS = ("standard", "relevant", "weak")


@dataclass
class StreamState:
    attempted_steps: int = 0
    committed_steps: int = 0
    rejected_steps: int = 0
    committed_edits: List[EditRequest] = field(default_factory=list)
    committed_bundles: List[ProbeBundle] = field(default_factory=list)
    committed_embeddings: List[List[float]] = field(default_factory=list)
    attempted_audit_probe_sets: List[ProbeSet] = field(default_factory=list)
    saver_snapshots: List[object] = field(default_factory=list)
    gate_scores: List[float] = field(default_factory=list)
    accepted_gate_scores: List[float] = field(default_factory=list)
    sampled_audit_risks: List[float] = field(default_factory=list)
    accepted_sampled_audit_risks: List[float] = field(default_factory=list)
    monitor_audit_pairs: List[tuple[float, float]] = field(default_factory=list)
    accepted_monitor_audit_pairs: List[tuple[float, float]] = field(default_factory=list)
    online_locality_prompt_count: int = 0
    audit_locality_prompt_count: int = 0
    probe_quality_info: Mapping[str, object] | None = None
    stopped_at: int | None = None
    stop_reason: str | None = None
    final_boundary_beta: float | None = None
    attempted_probe_bundles: List[ProbeBundle] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--events-output", type=pathlib.Path, default=None)
    parser.add_argument(
        "--checkpoints",
        default="100,500,1000,1500,2000,2500,5000",
        help="Comma-separated attempted-step checkpoints.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-edits", type=int, default=None)
    parser.add_argument(
        "--allow-short-stream",
        action="store_true",
        help="Allow running fewer than --min-edits when the prepared stream is short.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--theta", type=float, default=None)
    parser.add_argument("--target-acceptance", type=float, default=None)
    parser.add_argument("--gate-threshold", type=float, default=None)
    parser.add_argument("--gate-beta", type=float, default=None)
    parser.add_argument("--gate-history-size", type=int, default=None)
    parser.add_argument(
        "--probe-fraction",
        type=float,
        default=1.0,
        help="Deterministic nested fraction of online probes to use for monitoring.",
    )
    parser.add_argument(
        "--probe-design",
        choices=PROBE_DESIGNS,
        default="standard",
        help="Probe-quality condition. Non-standard designs split locality into online/audit banks.",
    )
    parser.add_argument(
        "--locality-monitor-fraction",
        type=float,
        default=0.5,
        help="Fraction of each edit's locality bank used online for relevant/weak probe designs.",
    )
    parser.add_argument(
        "--weak-bottom-quantile",
        type=float,
        default=0.25,
        help="Weak probes are drawn from the lowest-similarity donor quantile.",
    )
    parser.add_argument(
        "--min-locality-prompts",
        type=int,
        default=0,
        help="Filter the stream to edits with at least this many locality prompts before applying --limit.",
    )
    parser.add_argument(
        "--skip-weak-base-correct-filter",
        action="store_true",
        help="Do not require weak donor probes to be predicted correctly by the base model.",
    )
    parser.add_argument("--ppl-text-path", type=pathlib.Path, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config/data/checkpoints without loading the editor model.",
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
        fixed_beta=(float(config["fixed_beta"]) if config.get("fixed_beta") is not None else None),
    )


def _write_json_atomic(path: pathlib.Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def _append_jsonl(path: pathlib.Path | None, payload: Mapping[str, object]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _parse_checkpoints(raw: str, limit: int) -> List[int]:
    requested = set()
    for token in raw.replace(" ", "").split(","):
        if token:
            requested.add(int(token))
    requested.add(limit)
    return sorted(step for step in requested if 1 <= step <= limit)


def _slice_probe_set(probe_set: ProbeSet, fraction: float) -> ProbeSet:
    pairs = list(zip(probe_set.prompts, probe_set.targets))
    if fraction >= 1.0 or not pairs:
        return probe_set
    keep_count = max(1, min(len(pairs), math.ceil(len(pairs) * fraction)))
    kept = pairs[:keep_count]
    return ProbeSet(
        prompts=[prompt for prompt, _ in kept],
        targets=[target for _, target in kept],
    )


def _slice_pairs(pairs: Sequence[tuple[str, str]], fraction: float) -> list[tuple[str, str]]:
    if fraction >= 1.0 or not pairs:
        return list(pairs)
    keep_count = max(1, min(len(pairs), math.ceil(len(pairs) * fraction)))
    return list(pairs[:keep_count])


class FractionalCounterFactProbeGenerator(CounterFactProbeGenerator):
    """CounterFact probes with deterministic nested probe-fraction thinning."""

    def __init__(self, probe_fraction: float) -> None:
        if not 0.0 < probe_fraction <= 1.0:
            raise ValueError("probe_fraction must lie in (0, 1].")
        self.probe_fraction = float(probe_fraction)

    def build(self, edit_request: EditRequest) -> ProbeBundle:
        bundle = super().build(edit_request)
        if self.probe_fraction >= 1.0:
            return bundle
        return ProbeBundle(
            edit_request=bundle.edit_request,
            edit_prompt=bundle.edit_prompt,
            generality=_slice_probe_set(bundle.generality, self.probe_fraction),
            locality=_slice_probe_set(bundle.locality, self.probe_fraction),
        )


def _stable_seed(seed: int, *parts: object) -> int:
    payload = "|".join([str(seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _word_count(text: str) -> int:
    return len(re.findall(r"\w+", text))


def _entity_tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9]+", text)
        if len(token) > 2
    }


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _average_ranks(values: Sequence[float]) -> list[float]:
    ranked = sorted((float(value), index) for index, value in enumerate(values))
    ranks = [0.0] * len(ranked)
    cursor = 0
    while cursor < len(ranked):
        end = cursor + 1
        while end < len(ranked) and ranked[end][0] == ranked[cursor][0]:
            end += 1
        average_rank = (cursor + 1 + end) / 2.0
        for _, original_index in ranked[cursor:end]:
            ranks[original_index] = average_rank
        cursor = end
    return ranks


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_var = sum((x - x_mean) ** 2 for x in xs)
    y_var = sum((y - y_mean) ** 2 for y in ys)
    if x_var == 0.0 or y_var == 0.0:
        return None
    return numerator / math.sqrt(x_var * y_var)


def _spearman(pairs: Sequence[tuple[float, float]]) -> float | None:
    if len(pairs) < 2:
        return None
    xs = [pair[0] for pair in pairs]
    ys = [pair[1] for pair in pairs]
    return _pearson(_average_ranks(xs), _average_ranks(ys))


class ProbeQualityCounterFactProbeGenerator(CounterFactProbeGenerator):
    """Build matched relevant/weak online probes plus held-out relevant audit probes."""

    def __init__(
        self,
        edits: Sequence[EditRequest],
        *,
        condition: str,
        split_fraction: float,
        weak_bottom_quantile: float,
        probe_fraction: float,
        seed: int,
        embedding_dim: int,
        base_correct_keys: set[tuple[str, str]] | None = None,
    ) -> None:
        if condition not in {"relevant", "weak"}:
            raise ValueError("Probe-quality condition must be 'relevant' or 'weak'.")
        if not 0.0 < split_fraction < 1.0:
            raise ValueError("locality monitor fraction must lie in (0, 1).")
        if not 0.0 < weak_bottom_quantile <= 1.0:
            raise ValueError("weak bottom quantile must lie in (0, 1].")
        if not 0.0 < probe_fraction <= 1.0:
            raise ValueError("probe_fraction must lie in (0, 1].")
        self.condition = condition
        self.split_fraction = float(split_fraction)
        self.weak_bottom_quantile = float(weak_bottom_quantile)
        self.probe_fraction = float(probe_fraction)
        self.seed = int(seed)
        self.embedding_dim = int(embedding_dim)
        self.base_correct_keys = base_correct_keys
        self._source_to_bundle: dict[str, ProbeBundle] = {}
        self._source_to_split: dict[str, tuple[list[tuple[str, str]], list[tuple[str, str]]]] = {}
        self._source_to_index: dict[str, int] = {}
        self.weak_fallback_count = 0
        self.weak_requested_count = 0
        self.weak_returned_count = 0

        donor_pool: list[dict[str, object]] = []
        for index, edit in enumerate(edits):
            source_id = str(edit.metadata.get("source_id", index))
            bundle = super().build(edit)
            self._source_to_bundle[source_id] = bundle
            self._source_to_index[source_id] = index
            monitor_pairs, audit_pairs = self._split_locality(source_id, bundle.locality)
            self._source_to_split[source_id] = (monitor_pairs, audit_pairs)
            for prompt, target in zip(bundle.locality.prompts, bundle.locality.targets):
                donor_pool.append(
                    {
                        "source_id": source_id,
                        "subject": edit.subject,
                        "relation": edit.relation,
                        "target_new": edit.target,
                        "prompt": prompt,
                        "target": target,
                        "embedding": list(hashed_text_embedding(prompt, self.embedding_dim)),
                        "prompt_words": _word_count(prompt),
                        "target_words": _word_count(target),
                    }
                )
        self.donor_pool = donor_pool

    def _split_locality(
        self,
        source_id: str,
        locality: ProbeSet,
    ) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        pairs = list(zip(locality.prompts, locality.targets))
        if len(pairs) < 2:
            return pairs, []
        rng = random.Random(_stable_seed(self.seed, "probe-quality-split", source_id))
        shuffled = list(pairs)
        rng.shuffle(shuffled)
        monitor_count = int(round(len(shuffled) * self.split_fraction))
        monitor_count = max(1, min(len(shuffled) - 1, monitor_count))
        return shuffled[:monitor_count], shuffled[monitor_count:]

    def editor_bundle(self, edit_request: EditRequest) -> ProbeBundle:
        source_id = str(edit_request.metadata.get("source_id", ""))
        return self._source_to_bundle[source_id]

    def audit_locality(self, edit_request: EditRequest) -> ProbeSet:
        source_id = str(edit_request.metadata.get("source_id", ""))
        _, audit_pairs = self._source_to_split[source_id]
        return ProbeSet(
            prompts=[prompt for prompt, _ in audit_pairs],
            targets=[target for _, target in audit_pairs],
        )

    def _weak_locality(
        self,
        edit_request: EditRequest,
        monitor_pairs: Sequence[tuple[str, str]],
        audit_pairs: Sequence[tuple[str, str]],
    ) -> list[tuple[str, str]]:
        requested_count = len(monitor_pairs)
        self.weak_requested_count += requested_count
        if requested_count <= 0:
            return []

        source_id = str(edit_request.metadata.get("source_id", ""))
        current_bundle = self._source_to_bundle[source_id]
        current_embedding = list(hashed_text_embedding(current_bundle.edit_prompt, self.embedding_dim))
        current_subject_tokens = _entity_tokens(edit_request.subject)
        blocked_prompts = {prompt for prompt, _ in audit_pairs}
        blocked_prompts.update(prompt for prompt, _ in monitor_pairs)
        target_prompt_words = (
            sum(_word_count(prompt) for prompt, _ in monitor_pairs) / len(monitor_pairs)
            if monitor_pairs
            else _word_count(current_bundle.edit_prompt)
        )
        target_answer_words = (
            sum(_word_count(target) for _, target in monitor_pairs) / len(monitor_pairs)
            if monitor_pairs
            else 1.0
        )

        candidates: list[tuple[float, float, dict[str, object]]] = []
        relaxed_candidates: list[tuple[float, float, dict[str, object]]] = []
        for donor in self.donor_pool:
            prompt = str(donor["prompt"])
            target = str(donor["target"])
            if donor["source_id"] == source_id or prompt in blocked_prompts:
                continue
            if donor["subject"] == edit_request.subject:
                continue
            if donor["target_new"] == edit_request.target:
                continue
            similarity = _cosine(current_embedding, cast(Sequence[float], donor["embedding"]))
            length_gap = abs(float(donor["prompt_words"]) - target_prompt_words)
            length_gap += abs(float(donor["target_words"]) - target_answer_words)
            scored = (similarity, length_gap, donor)
            relaxed_candidates.append(scored)
            if current_subject_tokens & _entity_tokens(prompt):
                continue
            candidates.append(scored)

        candidates.sort(key=lambda item: (item[0], item[1], str(item[2]["prompt"])))
        relaxed_candidates.sort(key=lambda item: (item[0], item[1], str(item[2]["prompt"])))
        bottom_count = max(requested_count, int(math.ceil(len(candidates) * self.weak_bottom_quantile)))
        bottom_candidates = candidates[:bottom_count]

        if self.base_correct_keys is not None:
            base_filtered = [
                item
                for item in bottom_candidates
                if (str(item[2]["prompt"]), str(item[2]["target"])) in self.base_correct_keys
            ]
        else:
            base_filtered = bottom_candidates

        if len(base_filtered) < requested_count:
            self.weak_fallback_count += 1
            wider = [
                item
                for item in candidates
                if self.base_correct_keys is None
                or (str(item[2]["prompt"]), str(item[2]["target"])) in self.base_correct_keys
            ]
            if len(wider) >= requested_count:
                base_filtered = wider
            elif len(candidates) >= requested_count:
                base_filtered = candidates
            else:
                base_filtered = relaxed_candidates

        selected: list[tuple[str, str]] = []
        seen: set[str] = set()
        for _, _, donor in sorted(base_filtered, key=lambda item: (item[1], item[0], str(item[2]["prompt"]))):
            prompt = str(donor["prompt"])
            if prompt in seen:
                continue
            seen.add(prompt)
            selected.append((prompt, str(donor["target"])))
            if len(selected) >= requested_count:
                break

        self.weak_returned_count += len(selected)
        return selected

    def build(self, edit_request: EditRequest) -> ProbeBundle:
        source_id = str(edit_request.metadata.get("source_id", ""))
        base = self._source_to_bundle[source_id]
        monitor_pairs, audit_pairs = self._source_to_split[source_id]
        selected_monitor_pairs = _slice_pairs(monitor_pairs, self.probe_fraction)
        if self.condition == "weak":
            locality_pairs = self._weak_locality(edit_request, selected_monitor_pairs, audit_pairs)
        else:
            locality_pairs = list(selected_monitor_pairs)
        return ProbeBundle(
            edit_request=base.edit_request,
            edit_prompt=base.edit_prompt,
            generality=base.generality,
            locality=ProbeSet(
                prompts=[prompt for prompt, _ in locality_pairs],
                targets=[target for _, target in locality_pairs],
            ),
        )

    def summary(self) -> dict[str, object]:
        monitor_counts = []
        selected_monitor_counts = []
        audit_counts = []
        for monitor_pairs, audit_pairs in self._source_to_split.values():
            monitor_counts.append(len(monitor_pairs))
            selected_monitor_counts.append(len(_slice_pairs(monitor_pairs, self.probe_fraction)))
            audit_counts.append(len(audit_pairs))
        return {
            "condition": self.condition,
            "split_fraction": self.split_fraction,
            "probe_fraction": self.probe_fraction,
            "weak_bottom_quantile": self.weak_bottom_quantile,
            "donor_pool_size": len(self.donor_pool),
            "monitor_prompt_count_mean": _mean(monitor_counts),
            "selected_monitor_prompt_count_mean": _mean(selected_monitor_counts),
            "audit_prompt_count_mean": _mean(audit_counts),
            "weak_requested_count": self.weak_requested_count,
            "weak_returned_count": self.weak_returned_count,
            "weak_fallback_count": self.weak_fallback_count,
            "base_correct_filter_enabled": self.base_correct_keys is not None,
            "base_correct_donor_count": (
                len(self.base_correct_keys) if self.base_correct_keys is not None else None
            ),
        }


def _compute_base_correct_keys(
    *,
    generator: ProbeQualityCounterFactProbeGenerator,
    model: object,
    tokenizer: object,
    max_prompt_tokens: int,
) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    seen: set[tuple[str, str]] = set()
    for donor in generator.donor_pool:
        prompt = str(donor["prompt"])
        target = str(donor["target"])
        key = (prompt, target)
        if key in seen:
            continue
        seen.add(key)
        if first_token_exact_match(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            target_text=target,
            max_prompt_tokens=max_prompt_tokens,
        ):
            keys.add(key)
    return keys


def _empty_probe_set() -> ProbeSet:
    return ProbeSet(prompts=[], targets=[])


def _load_config(args: argparse.Namespace) -> dict:
    with args.config.open("r", encoding="utf-8") as handle:
        config = apply_env_editor_overrides(json.load(handle))
    if args.seed is not None:
        config["seed"] = int(args.seed)
    if args.theta is not None:
        config["theta"] = float(args.theta)
    return config


def _load_stream(
    config: dict,
    *,
    limit_override: int | None,
    min_edits_override: int | None,
    allow_short_stream: bool,
    min_locality_prompts: int = 0,
) -> List[EditRequest]:
    source_path = PROJECT_ROOT / config["dataset_path"]
    edits = load_counterfact_like_jsonl(source_path)
    if min_locality_prompts > 0:
        edits = [
            edit
            for edit in edits
            if len(edit.metadata.get("locality_prompts", [])) >= min_locality_prompts
        ]
    configured_limit = int(config.get("limit", len(edits)))
    limit = int(limit_override) if limit_override is not None else configured_limit
    min_edits = (
        int(min_edits_override)
        if min_edits_override is not None
        else int(config.get("min_required_edits", 0) or 0)
    )

    available = len(edits)
    if min_edits and available < min_edits and not allow_short_stream:
        raise SystemExit(
            f"Prepared stream {source_path} has {available} edits, below required {min_edits}. "
            "Prepare a larger stream first or pass --allow-short-stream for a diagnostic run."
        )
    if limit > available and not allow_short_stream:
        raise SystemExit(
            f"Requested limit {limit} exceeds prepared stream size {available} at {source_path}."
        )
    return edits[: min(limit, available)]


def _metric_success_count(audit: Mapping[str, object], rate_key: str, count_key: str) -> float | None:
    rate = audit.get(rate_key)
    count = audit.get(count_key)
    if rate is None or count is None:
        return None
    return float(rate) * float(count)


def _attempt_denominators(edits: Sequence[EditRequest]) -> dict[str, int]:
    return {
        "rewrite_prompt_count": len(edits),
        "paraphrase_prompt_count": sum(len(edit.paraphrases) for edit in edits),
        "portability_prompt_count": sum(
            len(edit.metadata.get("portability_prompts", [])) for edit in edits
        ),
        "locality_prompt_count": sum(
            len(edit.metadata.get("locality_prompts", [])) for edit in edits
        ),
    }


def _safe_divide(numerator: float | None, denominator: int) -> float | None:
    if numerator is None or denominator <= 0:
        return None
    return numerator / denominator


def _score_probe_sets_exact_match(
    *,
    model: object,
    tokenizer: object,
    probe_sets: Sequence[ProbeSet],
    max_prompt_tokens: int,
) -> dict[str, object]:
    total = 0
    success = 0
    for probe_set in probe_sets:
        for prompt, target in zip(probe_set.prompts, probe_set.targets):
            total += 1
            success += int(
                first_token_exact_match(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    target_text=target,
                    max_prompt_tokens=max_prompt_tokens,
                )
            )
    success_rate = success / total if total else None
    return {
        "audit_prompt_count": total,
        "audit_success_count": success,
        "audit_nsr": success_rate,
        "audit_exact_risk": (1.0 - success_rate) if success_rate is not None else None,
    }


def _current_model_proposal(adapter: EasyEditAdapter) -> EditorProposal:
    return EditorProposal(
        metadata={"method": adapter.method},
        handle=SimpleNamespace(
            edited_model=adapter.model,
            runtime_model=adapter.model,
            tokenizer=adapter.tokenizer,
        ),
    )


def _posthoc_beta(config: Mapping[str, object], state: StreamState) -> float:
    if state.final_boundary_beta is not None:
        return float(state.final_boundary_beta)
    if config.get("gate_beta") is not None:
        return float(config["gate_beta"])
    return max(float(value) for value in config["beta_grid"])


def _score_posthoc_risk(
    *,
    config: Mapping[str, object],
    state: StreamState,
    adapter: EasyEditAdapter,
    max_prompt_tokens: int,
) -> dict[str, object]:
    if not state.attempted_probe_bundles:
        return {
            "posthoc_beta": None,
            "posthoc_risk": None,
            "posthoc_generality_risk": None,
            "posthoc_locality_risk": None,
            "posthoc_generality_prompt_count": 0,
            "posthoc_locality_prompt_count": 0,
        }
    beta = _posthoc_beta(config, state)
    bundle = _merge_probe_bundle(
        state.attempted_probe_bundles[-1],
        state.attempted_probe_bundles[:-1],
        max_history=len(state.attempted_probe_bundles) - 1,
    )
    evaluation = FirstTokenCausalLMEvaluator(max_prompt_tokens=max_prompt_tokens).evaluate(
        proposal=_current_model_proposal(adapter),
        probe_bundle=bundle,
        beta_grid=[beta],
        locality_weight=float(config["locality_weight"]),
    )
    return {
        "posthoc_beta": beta,
        "posthoc_risk": evaluation.joint_risk[beta],
        "posthoc_generality_risk": evaluation.generality_risk[beta],
        "posthoc_locality_risk": evaluation.locality_risk[beta],
        "posthoc_generality_prompt_count": len(bundle.generality.prompts),
        "posthoc_locality_prompt_count": len(bundle.locality.prompts),
    }


def _monitoring_summary(state: StreamState, mode: str) -> dict[str, object]:
    if mode == "saver":
        accepted_sampled_risks: List[float] = []
        sampled_risks: List[float] = []
        for snapshot in state.saver_snapshots:
            chosen_beta = getattr(snapshot, "chosen_beta", None)
            oracle_risks = getattr(snapshot, "oracle_risks", {})
            if chosen_beta is None or chosen_beta not in oracle_risks:
                continue
            risk = float(oracle_risks[chosen_beta])
            sampled_risks.append(risk)
            if getattr(snapshot, "candidate_committed", False):
                accepted_sampled_risks.append(risk)
        summary = {
            "monitored_risk": _mean(accepted_sampled_risks),
            "sampled_candidate_risk": _mean(sampled_risks),
            "sampled_candidate_count": len(sampled_risks),
            "accepted_sampled_candidate_count": len(accepted_sampled_risks),
        }
        if state.sampled_audit_risks or state.accepted_sampled_audit_risks:
            summary.update(
                {
                    "audit_risk": _mean(state.accepted_sampled_audit_risks),
                    "sampled_audit_risk": _mean(state.sampled_audit_risks),
                    "monitor_audit_spearman": _spearman(state.accepted_monitor_audit_pairs),
                    "sampled_monitor_audit_spearman": _spearman(state.monitor_audit_pairs),
                    "monitor_audit_pair_count": len(state.accepted_monitor_audit_pairs),
                    "sampled_monitor_audit_pair_count": len(state.monitor_audit_pairs),
                }
            )
        return summary
    return {
        "monitored_risk": _mean(state.accepted_gate_scores),
        "sampled_candidate_risk": _mean(state.gate_scores),
        "sampled_candidate_count": len(state.gate_scores),
        "accepted_sampled_candidate_count": len(state.accepted_gate_scores),
    }


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    if not values:
        return None
    return sum(values) / len(values)


def _checkpoint_record(
    *,
    step: int,
    mode: str,
    config: Mapping[str, object],
    state: StreamState,
    attempted_edits: Sequence[EditRequest],
    adapter: EasyEditAdapter,
    max_prompt_tokens: int,
    ppl_texts: Sequence[str] | None,
) -> dict[str, object]:
    audit = score_counterfact_metrics(
        model=adapter.model,
        tokenizer=adapter.tokenizer,
        edits=state.committed_edits,
        max_prompt_tokens=max_prompt_tokens,
    )
    denominators = _attempt_denominators(attempted_edits)
    rewrite_success = _metric_success_count(audit, "esr", "rewrite_prompt_count")
    paraphrase_success = _metric_success_count(audit, "psr", "paraphrase_prompt_count")

    ppl_summary = {"ppl": None, "ppl_text_count": 0, "ppl_token_count": 0}
    if ppl_texts is not None:
        ppl_summary = dict(
            causal_lm_perplexity(
                model=adapter.model,
                tokenizer=adapter.tokenizer,
                texts=ppl_texts,
                max_length=max_prompt_tokens,
            )
        )

    record = {
        "step": step,
        "mode": mode,
        "attempted_steps": state.attempted_steps,
        "committed_steps": state.committed_steps,
        "rejected_steps": state.rejected_steps,
        "acceptance_rate": (
            state.committed_steps / state.attempted_steps if state.attempted_steps else 0.0
        ),
        "esr": audit["esr"],
        "all_esr": _safe_divide(rewrite_success, denominators["rewrite_prompt_count"]),
        "psr": audit["psr"],
        "all_psr": _safe_divide(paraphrase_success, denominators["paraphrase_prompt_count"]),
        "ptsr": audit["ptsr"],
        "nsr": audit["nsr"],
        "ppl": ppl_summary["ppl"],
        "ppl_text_count": ppl_summary["ppl_text_count"],
        "ppl_token_count": ppl_summary["ppl_token_count"],
        "final_boundary_beta": state.final_boundary_beta,
        "theta": float(config["theta"]),
        **_score_posthoc_risk(
            config=config,
            state=state,
            adapter=adapter,
            max_prompt_tokens=max_prompt_tokens,
        ),
        **_monitoring_summary(state, mode),
    }
    if state.attempted_audit_probe_sets:
        record.update(
            _score_probe_sets_exact_match(
                model=adapter.model,
                tokenizer=adapter.tokenizer,
                probe_sets=state.attempted_audit_probe_sets,
                max_prompt_tokens=max_prompt_tokens,
            )
        )
        record.update(
            {
                "online_locality_prompt_count": state.online_locality_prompt_count,
                "audit_locality_prompt_count": state.audit_locality_prompt_count,
            }
        )
    return record


def _merge_probe_bundle(
    current: ProbeBundle,
    history: Sequence[ProbeBundle],
    *,
    max_history: int,
) -> ProbeBundle:
    selected_history = list(history[-max_history:]) if max_history > 0 else []
    generality_pairs = list(zip(current.generality.prompts, current.generality.targets))
    locality_pairs = list(zip(current.locality.prompts, current.locality.targets))

    for bundle in selected_history:
        generality_pairs.extend(zip(bundle.generality.prompts, bundle.generality.targets))
        locality_pairs.extend(zip(bundle.locality.prompts, bundle.locality.targets))

    return ProbeBundle(
        edit_request=current.edit_request,
        edit_prompt=current.edit_prompt,
        generality=ProbeSet(
            prompts=[prompt for prompt, _ in generality_pairs],
            targets=[target for _, target in generality_pairs],
        ),
        locality=ProbeSet(
            prompts=[prompt for prompt, _ in locality_pairs],
            targets=[target for _, target in locality_pairs],
        ),
    )


def _bundle_prompts(bundle: ProbeBundle) -> List[str]:
    prompts = list(bundle.generality.prompts)
    prompts.extend(bundle.locality.prompts)
    return prompts


def _next_token_log_probs(
    *,
    model: object,
    tokenizer: object,
    prompts: Sequence[str],
    max_prompt_tokens: int,
) -> list[object]:
    import torch

    device = next(model.parameters()).device
    outputs = []
    model.eval()
    for prompt in prompts:
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_prompt_tokens,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            logits = model(**encoded).logits[0, -1, :].float()
            outputs.append(torch.log_softmax(logits, dim=-1).detach().cpu())
    return outputs


def _mean_kl(old_log_probs: Sequence[object], new_log_probs: Sequence[object]) -> float:
    import torch

    values = []
    for old_log, new_log in zip(old_log_probs, new_log_probs):
        values.append(torch.sum(torch.exp(old_log) * (old_log - new_log.cpu())).item())
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _resolve_gate_beta(config: Mapping[str, object], gate_beta: float | None) -> float:
    if gate_beta is not None:
        return float(gate_beta)
    configured = config.get("gate_beta")
    if configured is not None:
        return float(configured)
    return max(float(value) for value in config["beta_grid"])


def _resolve_gate_threshold(config: Mapping[str, object], gate_threshold: float | None) -> float:
    if gate_threshold is not None:
        return float(gate_threshold)
    configured = config.get("gate_threshold")
    if configured is not None:
        return float(configured)
    return float(config["theta"])


def _resolve_target_acceptance(config: Mapping[str, object], target_acceptance: float | None) -> float:
    if target_acceptance is not None:
        return float(target_acceptance)
    configured = config.get("target_acceptance")
    if configured is not None:
        return float(configured)
    raise SystemExit("--target-acceptance or config['target_acceptance'] is required for random_reject.")


def _resolve_gate_history_size(config: Mapping[str, object], gate_history_size: int | None) -> int:
    if gate_history_size is not None:
        return int(gate_history_size)
    return int(config.get("gate_history_size", 4))


def _matched_gate_accepts(
    *,
    gate_score: float,
    state: StreamState,
    target_acceptance: float | None,
    gate_threshold: float,
) -> bool:
    if gate_score > gate_threshold:
        return False
    if target_acceptance is None:
        return True
    desired_commits = round(target_acceptance * state.attempted_steps)
    if state.committed_steps >= desired_commits:
        return False
    accepted_rank_budget = max(1, math.ceil(target_acceptance * len(state.gate_scores)))
    rank = sum(1 for score in state.gate_scores if score <= gate_score)
    return rank <= accepted_rank_budget


def _build_adapter(config: Mapping[str, object], mode: str) -> EasyEditAdapter:
    editor_config = config["editor"]
    return EasyEditAdapter(
        method=editor_config["method"],
        hparams_path=PROJECT_ROOT / editor_config["hparams_path"],
        overrides=_editor_overrides(editor_config, mode=mode),
    )


def _run_stream(
    *,
    args: argparse.Namespace,
    config: dict,
    edits: Sequence[EditRequest],
    checkpoints: Sequence[int],
    adapter: EasyEditAdapter,
    ppl_texts: Sequence[str] | None,
) -> dict[str, object]:
    rng = random.Random(int(config["seed"]))
    evaluator = FirstTokenCausalLMEvaluator(max_prompt_tokens=int(config.get("max_prompt_tokens", 256)))
    state = StreamState()
    records: List[dict[str, object]] = []
    started_at = dt.datetime.now().astimezone()
    started_monotonic = time.perf_counter()
    max_prompt_tokens = int(config.get("max_prompt_tokens", 256))
    probe_quality_generator: ProbeQualityCounterFactProbeGenerator | None = None
    if args.probe_design == "standard":
        probe_generator: CounterFactProbeGenerator = FractionalCounterFactProbeGenerator(args.probe_fraction)
    else:
        probe_quality_generator = ProbeQualityCounterFactProbeGenerator(
            edits,
            condition=args.probe_design,
            split_fraction=float(args.locality_monitor_fraction),
            weak_bottom_quantile=float(args.weak_bottom_quantile),
            probe_fraction=float(args.probe_fraction),
            seed=int(config["seed"]),
            embedding_dim=int(config["embedding_dim"]),
        )
        if args.probe_design == "weak" and not args.skip_weak_base_correct_filter:
            base_correct_keys = _compute_base_correct_keys(
                generator=probe_quality_generator,
                model=adapter.model,
                tokenizer=adapter.tokenizer,
                max_prompt_tokens=max_prompt_tokens,
            )
            probe_quality_generator.base_correct_keys = base_correct_keys
        probe_generator = probe_quality_generator

    monitor = None
    tension_scorer = None
    embedding_dim = int(config["embedding_dim"])
    if args.mode == "saver":
        monitor = SaverMonitor(_build_saver_config(config))
        tension_scorer = StructuralTensionScorer(history_k=int(config["history_k"]))

    target_acceptance = None
    if args.mode in {"random_reject", "probe_gate", "kl_gate"} and args.target_acceptance is not None:
        target_acceptance = _resolve_target_acceptance(config, args.target_acceptance)
        if not 0.0 <= target_acceptance <= 1.0:
            raise SystemExit("target acceptance must lie in [0, 1].")
    elif args.mode == "random_reject":
        target_acceptance = _resolve_target_acceptance(config, args.target_acceptance)
        if not 0.0 <= target_acceptance <= 1.0:
            raise SystemExit("target acceptance must lie in [0, 1].")

    gate_beta = _resolve_gate_beta(config, args.gate_beta)
    gate_threshold = _resolve_gate_threshold(config, args.gate_threshold)
    gate_history_size = _resolve_gate_history_size(config, args.gate_history_size)
    checkpoint_set = set(checkpoints)

    def emit_checkpoint() -> None:
        if probe_quality_generator is not None:
            state.probe_quality_info = probe_quality_generator.summary()
        records.append(
            _checkpoint_record(
                step=state.attempted_steps,
                mode=args.mode,
                config=config,
                state=state,
                attempted_edits=edits[: state.attempted_steps],
                adapter=adapter,
                max_prompt_tokens=max_prompt_tokens,
                ppl_texts=ppl_texts,
            )
        )
        result = _result_payload(
            args=args,
            config=config,
            edits=edits,
            checkpoints=checkpoints,
            state=state,
            records=records,
            started_at=started_at,
            wall_clock_seconds=time.perf_counter() - started_monotonic,
        )
        _write_json_atomic(args.output, result)

    for edit in edits:
        if state.stopped_at is not None:
            break

        state.attempted_steps += 1
        probe_bundle = probe_generator.build(edit)
        state.attempted_probe_bundles.append(probe_bundle)
        editor_probe_bundle = (
            probe_quality_generator.editor_bundle(edit)
            if probe_quality_generator is not None
            else probe_bundle
        )
        audit_locality = (
            probe_quality_generator.audit_locality(edit)
            if probe_quality_generator is not None
            else _empty_probe_set()
        )
        if probe_quality_generator is not None:
            state.attempted_audit_probe_sets.append(audit_locality)
            state.online_locality_prompt_count += len(probe_bundle.locality.prompts)
            state.audit_locality_prompt_count += len(audit_locality.prompts)
        event: dict[str, object] = {
            "step": state.attempted_steps,
            "mode": args.mode,
            "source_id": edit.metadata.get("source_id"),
            "timestamp": dt.datetime.now().astimezone().isoformat(),
            "probe_design": args.probe_design,
        }
        if probe_quality_generator is not None:
            event.update(
                {
                    "online_locality_prompt_count": len(probe_bundle.locality.prompts),
                    "audit_locality_prompt_count": len(audit_locality.prompts),
                }
            )

        if args.mode == "random_reject":
            should_commit = rng.random() < float(target_acceptance)
            if should_commit:
                proposal = adapter.propose(editor_probe_bundle)
                adapter.commit(proposal)
                state.committed_steps += 1
                state.committed_edits.append(edit)
                state.committed_bundles.append(probe_bundle)
                event["decision"] = "committed"
            else:
                state.rejected_steps += 1
                event["decision"] = "rejected"

        elif args.mode == "unconstrained":
            proposal = adapter.propose(editor_probe_bundle)
            adapter.commit(proposal)
            state.committed_steps += 1
            state.committed_edits.append(edit)
            state.committed_bundles.append(probe_bundle)
            event["decision"] = "committed"

        elif args.mode == "probe_gate":
            proposal = adapter.propose(editor_probe_bundle)
            scoring_bundle = _merge_probe_bundle(
                probe_bundle,
                state.committed_bundles,
                max_history=gate_history_size,
            )
            evaluation = evaluator.evaluate(
                proposal=proposal,
                probe_bundle=scoring_bundle,
                beta_grid=[gate_beta],
                locality_weight=float(config["locality_weight"]),
            )
            gate_score = float(evaluation.joint_risk[gate_beta])
            state.gate_scores.append(gate_score)
            should_commit = _matched_gate_accepts(
                gate_score=gate_score,
                state=state,
                target_acceptance=target_acceptance,
                gate_threshold=gate_threshold,
            )
            if should_commit:
                adapter.commit(proposal)
                state.committed_steps += 1
                state.committed_edits.append(edit)
                state.committed_bundles.append(probe_bundle)
                state.accepted_gate_scores.append(gate_score)
                event["decision"] = "committed"
            else:
                adapter.rollback(proposal)
                state.rejected_steps += 1
                event["decision"] = "rejected"
            event["gate_score"] = gate_score
            event["gate_threshold"] = gate_threshold
            event["gate_beta"] = gate_beta
            event["target_acceptance"] = target_acceptance

        elif args.mode == "kl_gate":
            scoring_bundle = _merge_probe_bundle(
                probe_bundle,
                state.committed_bundles,
                max_history=gate_history_size,
            )
            prompts = _bundle_prompts(scoring_bundle)
            old_log_probs = _next_token_log_probs(
                model=adapter.model,
                tokenizer=adapter.tokenizer,
                prompts=prompts,
                max_prompt_tokens=max_prompt_tokens,
            )
            proposal = adapter.propose(editor_probe_bundle)
            runtime_model = getattr(proposal.handle, "runtime_model", adapter.model)
            new_log_probs = _next_token_log_probs(
                model=runtime_model,
                tokenizer=adapter.tokenizer,
                prompts=prompts,
                max_prompt_tokens=max_prompt_tokens,
            )
            gate_score = _mean_kl(old_log_probs, new_log_probs)
            state.gate_scores.append(gate_score)
            should_commit = _matched_gate_accepts(
                gate_score=gate_score,
                state=state,
                target_acceptance=target_acceptance,
                gate_threshold=gate_threshold,
            )
            if should_commit:
                adapter.commit(proposal)
                state.committed_steps += 1
                state.committed_edits.append(edit)
                state.committed_bundles.append(probe_bundle)
                state.accepted_gate_scores.append(gate_score)
                event["decision"] = "committed"
            else:
                adapter.rollback(proposal)
                state.rejected_steps += 1
                event["decision"] = "rejected"
            event["gate_score"] = gate_score
            event["gate_threshold"] = gate_threshold
            event["target_acceptance"] = target_acceptance

        else:
            assert args.mode == "saver"
            assert monitor is not None
            assert tension_scorer is not None
            proposal = adapter.propose(editor_probe_bundle)
            current_embedding = list(hashed_text_embedding(probe_bundle.edit_prompt, embedding_dim))
            structural_tension = tension_scorer.score(current_embedding, state.committed_embeddings)
            plan = monitor.plan_step(structural_tension=structural_tension, rng=rng)

            oracle_risks = None
            sampled_monitor_risk = None
            sampled_audit_risk = None
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
            state.saver_snapshots.append(snapshot)
            state.final_boundary_beta = snapshot.boundary_beta
            if plan.sampled and snapshot.chosen_beta is not None:
                chosen_beta = float(snapshot.chosen_beta)
                if oracle_risks is not None and chosen_beta in oracle_risks:
                    sampled_monitor_risk = float(oracle_risks[chosen_beta])
                if audit_locality.prompts:
                    audit_evaluation = evaluator.evaluate(
                        proposal=proposal,
                        probe_bundle=ProbeBundle(
                            edit_request=probe_bundle.edit_request,
                            edit_prompt=probe_bundle.edit_prompt,
                            generality=_empty_probe_set(),
                            locality=audit_locality,
                        ),
                        beta_grid=[chosen_beta],
                        locality_weight=1.0,
                    )
                    sampled_audit_risk = float(audit_evaluation.locality_risk[chosen_beta])
            if sampled_monitor_risk is not None:
                event["monitor_risk"] = sampled_monitor_risk
            if sampled_audit_risk is not None:
                event["audit_risk"] = sampled_audit_risk
                state.sampled_audit_risks.append(sampled_audit_risk)
            if sampled_monitor_risk is not None and sampled_audit_risk is not None:
                state.monitor_audit_pairs.append((sampled_monitor_risk, sampled_audit_risk))

            if snapshot.candidate_rejected:
                adapter.rollback(proposal)
                state.rejected_steps += 1
                event["decision"] = "rejected"
            else:
                adapter.commit(proposal)
                snapshot.candidate_committed = True
                monitor.accept(snapshot)
                state.committed_steps += 1
                state.committed_edits.append(edit)
                state.committed_bundles.append(probe_bundle)
                state.committed_embeddings.append(current_embedding)
                state.final_boundary_beta = monitor.boundary_beta
                event["decision"] = "committed"
                if sampled_audit_risk is not None:
                    state.accepted_sampled_audit_risks.append(sampled_audit_risk)
                if sampled_monitor_risk is not None and sampled_audit_risk is not None:
                    state.accepted_monitor_audit_pairs.append((sampled_monitor_risk, sampled_audit_risk))

            if monitor.boundary_saturated(snapshot):
                snapshot.stop_triggered = True
                snapshot.stop_reason = "boundary_evidence_exhausted"
                state.stopped_at = state.attempted_steps
                state.stop_reason = snapshot.stop_reason
            if monitor.config.rejection_policy == "stop" and snapshot.candidate_rejected:
                snapshot.stop_triggered = True
                snapshot.stop_reason = "rejected_edit"
                state.stopped_at = state.attempted_steps
                state.stop_reason = snapshot.stop_reason

            event.update(
                {
                    "sampled": snapshot.sampled,
                    "chosen_beta": snapshot.chosen_beta,
                    "boundary_beta": snapshot.boundary_beta,
                    "q_t": snapshot.q_t,
                }
            )

        event["committed_steps"] = state.committed_steps
        event["rejected_steps"] = state.rejected_steps
        event["acceptance_rate"] = (
            state.committed_steps / state.attempted_steps if state.attempted_steps else 0.0
        )
        _append_jsonl(args.events_output, event)

        if state.attempted_steps in checkpoint_set:
            emit_checkpoint()

    if not records or records[-1]["step"] != state.attempted_steps:
        emit_checkpoint()
    if probe_quality_generator is not None:
        state.probe_quality_info = probe_quality_generator.summary()

    return _result_payload(
        args=args,
        config=config,
        edits=edits,
        checkpoints=checkpoints,
        state=state,
        records=records,
        started_at=started_at,
        wall_clock_seconds=time.perf_counter() - started_monotonic,
    )


def _result_payload(
    *,
    args: argparse.Namespace,
    config: Mapping[str, object],
    edits: Sequence[EditRequest],
    checkpoints: Sequence[int],
    state: StreamState,
    records: Sequence[Mapping[str, object]],
    started_at: dt.datetime,
    wall_clock_seconds: float,
) -> dict[str, object]:
    return {
        "mode": args.mode,
        "config_path": str(args.config),
        "dataset_path": config["dataset_path"],
        "limit": len(edits),
        "min_required_edits": config.get("min_required_edits"),
        "checkpoints": list(checkpoints),
        "theta": float(config["theta"]),
        "alpha": float(config["alpha"]),
        "seed": int(config["seed"]),
        "probe_fraction": float(args.probe_fraction),
        "probe_design": args.probe_design,
        "locality_monitor_fraction": float(args.locality_monitor_fraction),
        "weak_bottom_quantile": float(args.weak_bottom_quantile),
        "min_locality_prompts": int(args.min_locality_prompts),
        "probe_quality_info": state.probe_quality_info,
        "beta_grid": [float(value) for value in config["beta_grid"]],
        "run_summary": {
            "attempted_steps": state.attempted_steps,
            "committed_steps": state.committed_steps,
            "rejected_steps": state.rejected_steps,
            "stopped_at": state.stopped_at,
            "stop_reason": state.stop_reason,
            "acceptance_rate": (
                state.committed_steps / state.attempted_steps if state.attempted_steps else 0.0
            ),
            "final_boundary_beta": state.final_boundary_beta,
        },
        "records": list(records),
        "timing": {
            "started_at": started_at.isoformat(),
            "updated_at": dt.datetime.now().astimezone().isoformat(),
            "wall_clock_seconds": wall_clock_seconds,
        },
        "editor_runtime": {
            "method": config["editor"]["method"],
            "resolved_overrides": _editor_overrides(config["editor"], mode=args.mode),
        },
        "notes": {
            "metric_unit": "first_token_exact_match",
            "all_esr_all_psr": "Rejected requests are assigned zero success.",
            "matched_acceptance": "Probe/KL gates use an online prefix-rank controller when --target-acceptance is supplied.",
            "beta_plot": "not_requested_for_rebuttal_stream",
            "probe_quality": "Non-standard probe designs split locality probes into online monitor and held-out audit banks.",
        },
    }


def _dry_run_payload(
    *,
    args: argparse.Namespace,
    config: Mapping[str, object],
    edits: Sequence[EditRequest],
    checkpoints: Sequence[int],
    ppl_texts: Sequence[str] | None,
) -> dict[str, object]:
    return {
        "dry_run": True,
        "mode": args.mode,
        "config_path": str(args.config),
        "dataset_path": config["dataset_path"],
        "available_loaded_edits": len(edits),
        "checkpoints": list(checkpoints),
        "seed": int(config["seed"]),
        "theta": float(config["theta"]),
        "editor_method": config["editor"]["method"],
        "probe_fraction": float(args.probe_fraction),
        "probe_design": args.probe_design,
        "locality_monitor_fraction": float(args.locality_monitor_fraction),
        "weak_bottom_quantile": float(args.weak_bottom_quantile),
        "min_locality_prompts": int(args.min_locality_prompts),
        "ppl_text_count": len(ppl_texts or []),
    }


def main() -> None:
    args = parse_args()
    config = _load_config(args)
    edits = _load_stream(
        config,
        limit_override=args.limit,
        min_edits_override=args.min_edits,
        allow_short_stream=args.allow_short_stream,
        min_locality_prompts=args.min_locality_prompts,
    )
    checkpoints = _parse_checkpoints(args.checkpoints, len(edits))

    ppl_text_path = args.ppl_text_path
    if ppl_text_path is None and config.get("ppl_text_path"):
        ppl_text_path = PROJECT_ROOT / str(config["ppl_text_path"])
    ppl_texts = load_ppl_texts(ppl_text_path) if ppl_text_path is not None else None

    if args.dry_run:
        result = _dry_run_payload(
            args=args,
            config=config,
            edits=edits,
            checkpoints=checkpoints,
            ppl_texts=ppl_texts,
        )
        _write_json_atomic(args.output, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    adapter = _build_adapter(config, args.mode)
    result = _run_stream(
        args=args,
        config=config,
        edits=edits,
        checkpoints=checkpoints,
        adapter=adapter,
        ppl_texts=ppl_texts,
    )
    _write_json_atomic(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
