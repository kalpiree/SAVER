from __future__ import annotations

import hashlib
import json
import math
import random
import re
import string
from pathlib import Path

from saver.core.monitor import SaverMonitor
from saver.core.proxy import StructuralTensionScorer
from saver.core.text_embedding import SentenceEmbedding
from saver.eval.first_token import FirstTokenCausalLMEvaluator
from scripts.run_sequential_editing import _build_saver_config, _editor_overrides


ROOT = Path(__file__).resolve().parents[1]


def read_rows(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    else:
        rows = json.loads(path.read_text())
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("Expected a list of records.")
    return rows


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def append_row(path, row):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def load_config(path, seed):
    from saver.runtime_config import apply_env_editor_overrides

    protocol = json.loads(Path(path).read_text())
    config = apply_env_editor_overrides(json.loads((ROOT / protocol["base_config"]).read_text()))
    config.update(protocol.get("monitor_overrides", {}))
    config["seed"] = int(seed)
    if config.get("rejection_policy") != "continue" or config.get("stop_on_boundary_saturation"):
        raise ValueError("These runs require continuation after rejection.")
    if config.get("hard_gate_sampled_risk") or config.get("monotone_beta_search"):
        raise ValueError("These runs require unrestricted monitor-based admission.")
    if config.get("boundary_policy", "adaptive") != "adaptive":
        raise ValueError("These runs require adaptive boundary selection.")
    beta = float(protocol["fixed_beta"])
    if not 0.0 < beta < 1.0:
        raise ValueError("fixed_beta must lie in (0, 1).")
    if protocol["answer_mode"] not in {"teacher_forced_exact", "greedy_exact"}:
        raise ValueError("Invalid answer_mode.")
    return config, protocol


def make_adapter(config, mode):
    import importlib
    import numpy as np
    import torch
    from saver.editors.easyedit import EasyEditAdapter

    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
    class TransactionalAdapter(EasyEditAdapter):
        def _snapshot_internal_state(self):
            if self.method != "alphaedit":
                return super()._snapshot_internal_state()
            module = importlib.import_module(self.apply_algo.__module__)
            if not hasattr(module, "cache_c") or not hasattr(module, "cache_c_new"):
                raise RuntimeError("AlphaEdit does not expose its covariance cache.")
            cache = module.cache_c
            return {"cache_c": cache.clone() if cache is not None else None, "cache_c_new": module.cache_c_new}

        def _restore_internal_state(self, state):
            if self.method != "alphaedit":
                return super()._restore_internal_state(state)
            module = importlib.import_module(self.apply_algo.__module__)
            module.cache_c = state["cache_c"]
            module.cache_c_new = state["cache_c_new"]

    return TransactionalAdapter(
        config["editor"]["method"], ROOT / config["editor"]["hparams_path"],
        _editor_overrides(config["editor"], mode=mode),
    )


def normalize_answer(text):
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def target_tokens(tokenizer, text):
    ids = tokenizer(" " + text.strip(), add_special_tokens=False).input_ids
    if not ids or not text.strip():
        raise ValueError("Empty target answer.")
    return list(ids)


def inputs_for(model, tokenizer, prompt, max_prompt_tokens):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_prompt_tokens)
    device = model.get_input_embeddings().weight.device
    return {name: value.to(device) for name, value in inputs.items()}


def generate(model, tokenizer, prompt, max_prompt_tokens, max_new_tokens, first_line=True):
    import torch

    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive.")
    model.eval()
    inputs = inputs_for(model, tokenizer, prompt, max_prompt_tokens)
    with torch.inference_mode():
        output = model.generate(
            **inputs, do_sample=False, num_beams=1, max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        )
    ids = output[0, inputs["input_ids"].shape[1]:].tolist()
    eos = getattr(model.generation_config, "eos_token_id", None)
    stops = set(eos if isinstance(eos, list) else [eos]) | {tokenizer.eos_token_id}
    completed = False
    kept = []
    for token in ids:
        if token in stops:
            completed = True
            break
        kept.append(token)
        decoded = tokenizer.decode(kept, skip_special_tokens=True).lstrip()
        if first_line and "\n" in decoded and decoded.split("\n", 1)[0].strip():
            completed = True
            break
    decoded = tokenizer.decode(kept, skip_special_tokens=True).lstrip()
    text = decoded.split("\n", 1)[0].strip() if first_line else decoded.strip()
    if first_line and completed:
        while kept and tokenizer.decode(kept[:-1], skip_special_tokens=True).strip() == text:
            kept.pop()
    return {"text": text, "token_ids": kept, "completed": completed}


def score_answer(model, tokenizer, prompt, target, protocol, max_prompt_tokens, token_ids=None):
    import torch

    ids = list(token_ids) if token_ids is not None else target_tokens(tokenizer, target)
    if not ids or len(ids) > int(protocol["max_answer_tokens"]):
        raise ValueError("Target is empty or exceeds max_answer_tokens.")
    model.eval()
    inputs = inputs_for(model, tokenizer, prompt, max_prompt_tokens)
    prompt_length = inputs["input_ids"].shape[1]
    prefix = torch.tensor([ids[:-1]], device=inputs["input_ids"].device, dtype=inputs["input_ids"].dtype)
    inputs["input_ids"] = torch.cat([inputs["input_ids"], prefix], dim=1)
    inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
    with torch.inference_mode():
        logits = model(**inputs).logits[0, prompt_length - 1:prompt_length + len(ids) - 1, :]
    probabilities = torch.softmax(logits[0].float(), dim=-1)
    sorted_probabilities, sorted_ids = torch.sort(probabilities, descending=True)
    rank = int((sorted_ids == ids[0]).nonzero(as_tuple=False)[0].item())
    cumulative = torch.cumsum(sorted_probabilities, dim=0)
    cutoff = int(torch.searchsorted(cumulative, cumulative.new_tensor(protocol["fixed_beta"])).item())
    correct = bool((logits.argmax(dim=-1) == torch.tensor(ids, device=logits.device)).all().item())
    if protocol["answer_mode"] == "greedy_exact":
        prediction = generate(model, tokenizer, prompt, max_prompt_tokens, protocol["max_answer_tokens"])
        correct = prediction["completed"] and normalize_answer(prediction["text"]) == normalize_answer(target)
    return {"correct": correct, "risk": float(rank > cutoff), "token_count": len(ids)}


def freeze_audit(model, tokenizer, rows, protocol, max_prompt_tokens):
    references = []
    excluded = []
    for row in rows:
        prediction = generate(model, tokenizer, row["prompt"], max_prompt_tokens, protocol["max_answer_tokens"])
        if not prediction["completed"] or not prediction["text"] or not prediction["token_ids"]:
            excluded.append(row["id"])
            continue
        references.append({
            "id": row["id"], "prompt": row["prompt"], "target": prediction["text"],
            "token_ids": prediction["token_ids"],
            "multi_token": len(target_tokens(tokenizer, prediction["text"])) >= 2,
        })
    if not references:
        raise ValueError("No complete initial-model locality answers within max_answer_tokens.")
    return references, excluded


def audit_scores(model, tokenizer, references, protocol, max_prompt_tokens):
    scored = [
        (row, score_answer(model, tokenizer, row["prompt"], row["target"], protocol, max_prompt_tokens, row["token_ids"]))
        for row in references
    ]
    multi = [score for row, score in scored if row["multi_token"]]
    accuracy = sum(int(score["correct"]) for _, score in scored) / len(scored)
    return {
        "accuracy": accuracy,
        "multi_accuracy": sum(int(score["correct"]) for score in multi) / len(multi) if multi else None,
        "multi_risk": sum(score["risk"] for score in multi) / len(multi) if multi else None,
        "multi_count": len(multi),
    }


class Controller:
    def __init__(self, config, edits, references, mode, embedding=None):
        self.config = config
        self.mode = mode
        self.monitor = SaverMonitor(_build_saver_config(config)) if mode == "saver" else None
        self.embedding = (embedding or SentenceEmbedding([edit.metadata["rewrite_prompt"] for edit in edits], config)) if self.monitor else None
        self.scorer = StructuralTensionScorer(config["history_k"])
        self.rng = random.Random(config["seed"])
        self.history = []
        self.evaluator = FirstTokenCausalLMEvaluator(config.get("max_prompt_tokens", 256), references)

    def pre_ratio(self):
        if self.monitor is None:
            return None
        values = [state.martingale for state in self.monitor.states.values()]
        ratio = min(values) / self.monitor.threshold
        if not math.isfinite(ratio):
            raise ValueError("Non-finite monitor statistic.")
        return ratio

    def decide(self, proposal, bundle):
        if self.monitor is None:
            return True, None, None
        vector = self.embedding(bundle.edit_prompt)
        tension = self.scorer.score(vector, self.history)
        plan = self.monitor.plan_step(tension, rng=self.rng)
        risks = None
        if plan.sampled:
            risks = self.evaluator.evaluate(
                proposal, bundle, self.monitor.beta_grid, self.config["locality_weight"],
            ).joint_risk
        snapshot = self.monitor.evaluate_candidate(plan, risks)
        self.monitor.observe_attempt(snapshot)
        if not snapshot.candidate_rejected:
            self.monitor.accept(snapshot)
            self.history.append(vector)
        return not snapshot.candidate_rejected, snapshot, self.pre_ratio()


def generate_biographies(adapter, rows, path, config, protocol):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for row in rows:
        result = generate(
            adapter.model, adapter.tokenizer, row["prompt"], config["max_prompt_tokens"],
            protocol["biography_max_new_tokens"], first_line=False,
        )
        append_row(path, {
            "id": row["id"], "topic": row["topic"], "prompt": row["prompt"],
            "generation": result["text"], "completed": result["completed"],
        })
