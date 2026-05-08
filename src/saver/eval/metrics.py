"""Final-model metric helpers for sequential editing experiments."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

from saver.types import EditRequest


def _lazy_torch():
    import torch

    return torch


def _model_device(model: object):
    return next(model.parameters()).device


def first_target_id(tokenizer: object, target_text: str) -> int:
    prefixed = tokenizer(" " + target_text, add_special_tokens=False).input_ids
    if prefixed:
        return int(prefixed[0])

    plain = tokenizer(target_text, add_special_tokens=False).input_ids
    if plain:
        return int(plain[0])
    raise ValueError(f"Could not tokenize target text '{target_text}'.")


def first_token_exact_match(
    model: object,
    tokenizer: object,
    prompt: str,
    target_text: str,
    max_prompt_tokens: int = 256,
) -> bool:
    torch = _lazy_torch()
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_tokens,
    )
    device = _model_device(model)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
        predicted_id = int(torch.argmax(logits).item())

    return predicted_id == first_target_id(tokenizer, target_text)


def score_counterfact_metrics(
    model: object,
    tokenizer: object,
    edits: Sequence[EditRequest],
    max_prompt_tokens: int = 256,
) -> Dict[str, object]:
    rewrite_total = 0
    rewrite_success = 0
    paraphrase_total = 0
    paraphrase_success = 0
    portability_total = 0
    portability_success = 0
    locality_total = 0
    locality_success = 0
    per_edit: List[Dict[str, object]] = []

    model.eval()

    for edit in edits:
        metadata = edit.metadata
        rewrite_prompt = metadata["rewrite_prompt"]
        rewrite_ok = first_token_exact_match(
            model=model,
            tokenizer=tokenizer,
            prompt=rewrite_prompt,
            target_text=edit.target,
            max_prompt_tokens=max_prompt_tokens,
        )
        rewrite_total += 1
        rewrite_success += int(rewrite_ok)

        paraphrases = list(edit.paraphrases)
        paraphrase_hits = 0
        for prompt in paraphrases:
            ok = first_token_exact_match(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                target_text=edit.target,
                max_prompt_tokens=max_prompt_tokens,
            )
            paraphrase_total += 1
            paraphrase_success += int(ok)
            paraphrase_hits += int(ok)

        portability_prompts = list(metadata.get("portability_prompts", []))
        portability_answers = list(metadata.get("portability_answers", []))
        portability_hits = 0
        for prompt, target in zip(portability_prompts, portability_answers):
            ok = first_token_exact_match(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                target_text=target,
                max_prompt_tokens=max_prompt_tokens,
            )
            portability_total += 1
            portability_success += int(ok)
            portability_hits += int(ok)

        locality_prompts = list(metadata.get("locality_prompts", []))
        locality_answers = list(metadata.get("locality_answers", []))
        locality_hits = 0
        for prompt, target in zip(locality_prompts, locality_answers):
            ok = first_token_exact_match(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                target_text=target,
                max_prompt_tokens=max_prompt_tokens,
            )
            locality_total += 1
            locality_success += int(ok)
            locality_hits += int(ok)

        per_edit.append(
            {
                "source_id": metadata.get("source_id"),
                "subject": edit.subject,
                "relation": edit.relation,
                "target": edit.target,
                "rewrite_success": rewrite_ok,
                "paraphrase_prompt_count": len(paraphrases),
                "paraphrase_success_rate": (
                    paraphrase_hits / len(paraphrases) if paraphrases else None
                ),
                "portability_prompt_count": len(portability_prompts),
                "portability_success_rate": (
                    portability_hits / len(portability_prompts) if portability_prompts else None
                ),
                "locality_prompt_count": len(locality_prompts),
                "locality_success_rate": (
                    locality_hits / len(locality_prompts) if locality_prompts else None
                ),
            }
        )

    return {
        "audit_scope": "committed_edits_only",
        "num_committed_edits": len(edits),
        "rewrite_prompt_count": rewrite_total,
        "paraphrase_prompt_count": paraphrase_total,
        "portability_prompt_count": portability_total,
        "locality_prompt_count": locality_total,
        "esr": (rewrite_success / rewrite_total) if rewrite_total else None,
        "psr": (paraphrase_success / paraphrase_total) if paraphrase_total else None,
        "ptsr": (portability_success / portability_total) if portability_total else None,
        "nsr": (locality_success / locality_total) if locality_total else None,
        "per_edit": per_edit,
    }


def load_ppl_texts(path: str | Path) -> List[str]:
    source = Path(path)
    suffix = source.suffix.lower()

    if suffix == ".txt":
        return [line.strip() for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]

    if suffix == ".jsonl":
        texts: List[str] = []
        with source.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                for key in ("text", "prompt", "rewrite_prompt"):
                    value = payload.get(key)
                    if isinstance(value, str) and value.strip():
                        texts.append(value.strip())
                        break
        return texts

    raise ValueError(f"Unsupported PPL text file format: {source}")


def causal_lm_perplexity(
    model: object,
    tokenizer: object,
    texts: Iterable[str],
    max_length: int = 256,
) -> Mapping[str, float | int | None]:
    torch = _lazy_torch()
    device = _model_device(model)
    model.eval()

    total_nll = 0.0
    total_predicted_tokens = 0
    text_count = 0

    for text in texts:
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        input_ids = encoded["input_ids"]
        seq_len = int(input_ids.shape[1])
        if seq_len < 2:
            continue

        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            outputs = model(**encoded, labels=encoded["input_ids"])
            loss = float(outputs.loss.item())

        predicted_tokens = seq_len - 1
        total_nll += loss * predicted_tokens
        total_predicted_tokens += predicted_tokens
        text_count += 1

    if total_predicted_tokens == 0:
        return {
            "ppl": None,
            "ppl_text_count": text_count,
            "ppl_token_count": total_predicted_tokens,
        }

    return {
        "ppl": math.exp(total_nll / total_predicted_tokens),
        "ppl_text_count": text_count,
        "ppl_token_count": total_predicted_tokens,
    }
