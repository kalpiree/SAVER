#!/usr/bin/env python3
"""Prepare a larger CounterFact stream from the Hugging Face `azhx/counterfact` dataset."""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
from typing import Any, Iterable, List

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--dataset", default="azhx/counterfact")
    parser.add_argument("--splits", default="train,test")
    parser.add_argument("--max-paraphrases", type=int, default=3)
    parser.add_argument("--max-locality", type=int, default=4)
    return parser.parse_args()


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        for key in ("str", "text", "answer", "label", "target", "ground_truth"):
            if key in value:
                return _text(value[key])
        return ""
    if isinstance(value, (list, tuple)):
        return _text(value[0]) if value else ""
    return str(value).strip()


def _ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _dedupe(values: Iterable[Any], limit: int | None = None) -> List[str]:
    output: List[str] = []
    seen: set[str] = set()
    for value in values:
        text = _text(value)
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
        if limit is not None and len(output) >= limit:
            break
    return output


def _format_prompt(prompt: str, subject: str) -> str:
    if "{}" not in prompt:
        return prompt
    try:
        return prompt.format(subject)
    except Exception:
        return prompt


def _locality_pairs(record: dict[str, Any], limit: int) -> tuple[List[str], List[str]]:
    neighborhood = record.get("neighborhood_prompts")
    prompts: List[str] = []
    answers: List[str] = []
    for item in _ensure_list(neighborhood):
        if isinstance(item, dict):
            prompt = _text(item.get("prompt") or item.get("src") or item.get("text"))
            answer = _text(
                item.get("target")
                or item.get("ground_truth")
                or item.get("answer")
                or item.get("target_true")
            )
        else:
            prompt = _text(item)
            answer = _text(record.get("target_true") or record.get("ground_truth"))
        if prompt and answer:
            prompts.append(prompt)
            answers.append(answer)
        if len(prompts) >= limit:
            break
    return prompts, answers


def _convert_record(record: dict[str, Any], index: int, args: argparse.Namespace) -> dict[str, Any] | None:
    requested = record.get("requested_rewrite")
    if isinstance(requested, list):
        requested = requested[0] if requested else None
    requested = requested if isinstance(requested, dict) else {}

    subject = _text(requested.get("subject") or record.get("subject"))
    prompt = _text(requested.get("prompt") or record.get("prompt") or record.get("src"))
    if prompt:
        prompt = _format_prompt(prompt, subject)
    target = _text(requested.get("target_new") or record.get("target_new") or record.get("target"))
    ground_truth = _text(
        requested.get("target_true")
        or record.get("target_true")
        or record.get("ground_truth")
    )
    paraphrases = _dedupe(
        record.get("paraphrase_prompts")
        or record.get("paraphrases")
        or record.get("rephrase_prompt"),
        limit=args.max_paraphrases,
    )
    locality_prompts, locality_answers = _locality_pairs(record, args.max_locality)

    if not prompt or not target:
        return None
    return {
        "id": str(record.get("case_id") or record.get("id") or f"hf-counterfact:{index}"),
        "subject": subject or prompt,
        "relation": _text(requested.get("relation") or record.get("relation")) or "counterfact",
        "target": target,
        "prompt": prompt,
        "ground_truth": ground_truth,
        "paraphrases": paraphrases,
        "locality_prompts": locality_prompts,
        "locality_answers": locality_answers,
    }


def main() -> None:
    args = parse_args()
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install `datasets` before preparing CounterFact.") from exc

    converted: List[dict[str, Any]] = []
    for split in [token.strip() for token in args.splits.split(",") if token.strip()]:
        dataset = load_dataset(args.dataset, split=split)
        for record in dataset:
            item = _convert_record(dict(record), len(converted), args)
            if item is not None:
                converted.append(item)

    rng = random.Random(args.seed)
    rng.shuffle(converted)
    if args.limit:
        converted = converted[: args.limit]

    output = args.output
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for item in converted:
            handle.write(json.dumps(item, ensure_ascii=True, sort_keys=True) + "\n")

    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "splits": args.splits,
                "output": str(output),
                "written_records": len(converted),
                "seed": args.seed,
                "limit": args.limit,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
