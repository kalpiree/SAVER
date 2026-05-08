#!/usr/bin/env python3
"""Convert raw CounterFact/zsRE/MQuAKE-style data into SAVER JSONL streams."""

from __future__ import annotations

import argparse
import json
import pathlib
import random
from typing import Any, Iterable, List, Sequence


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=pathlib.Path,
        required=True,
        help="Input .json or .jsonl dataset file.",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        required=True,
        help="Output SAVER-compatible JSONL path.",
    )
    parser.add_argument(
        "--format",
        choices=("auto", "counterfact", "zsre", "mquake"),
        default="auto",
        help="Input dataset family.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of converted records after shuffling/offset.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip this many converted records after shuffling.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle converted records before applying offset/limit.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help="Random seed used with --shuffle.",
    )
    parser.add_argument(
        "--max-paraphrases",
        type=int,
        default=3,
        help="Maximum paraphrase prompts to keep per record.",
    )
    parser.add_argument(
        "--max-locality",
        type=int,
        default=4,
        help="Maximum locality prompts to keep per record.",
    )
    parser.add_argument(
        "--max-portability",
        type=int,
        default=5,
        help="Maximum portability/compositional prompts to keep per record.",
    )
    return parser.parse_args()


def _load_records(path: pathlib.Path) -> List[dict[str, Any]]:
    if path.suffix == ".jsonl":
        records: List[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "records", "examples"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise ValueError(f"Unsupported dataset container in {path}.")


def _ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        for key in ("str", "text", "answer", "label", "target", "ground_truth"):
            if key in value:
                return _text(value[key])
        return ""
    if isinstance(value, (list, tuple)):
        if not value:
            return ""
        return _text(value[0])
    return str(value).strip()


def _dedupe_nonempty(values: Iterable[Any], limit: int | None = None) -> List[str]:
    deduped: List[str] = []
    seen: set[str] = set()
    for value in values:
        text = _text(value)
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append(text)
        if limit is not None and len(deduped) >= limit:
            break
    return deduped


def _format_prompt(prompt: str, subject: str) -> str:
    if "{}" not in prompt:
        return prompt
    try:
        return prompt.format(subject)
    except Exception:
        return prompt


def _pair_lists(prompts: Sequence[Any], answers: Sequence[Any], limit: int) -> tuple[List[str], List[str]]:
    prompt_texts = _dedupe_nonempty(prompts)
    answer_texts = [_text(value) for value in answers if _text(value)]
    if prompt_texts and len(answer_texts) == 1 and len(prompt_texts) > 1:
        answer_texts = answer_texts * len(prompt_texts)
    count = min(len(prompt_texts), len(answer_texts), limit)
    return prompt_texts[:count], answer_texts[:count]


def _extract_locality_pairs(record: dict[str, Any], max_locality: int) -> tuple[List[str], List[str]]:
    locality_block = record.get("locality")
    if isinstance(locality_block, dict):
        prompts: List[str] = []
        answers: List[str] = []
        for bucket in locality_block.values():
            for item in _ensure_list(bucket):
                if not isinstance(item, dict):
                    continue
                prompt = _text(item.get("prompt"))
                answer = _text(item.get("ground_truth"))
                if prompt and answer:
                    prompts.append(prompt)
                    answers.append(answer)
                if len(prompts) >= max_locality:
                    return prompts, answers
        if prompts:
            return prompts[:max_locality], answers[:max_locality]

    if "neighborhood_prompts" in record:
        prompts: List[str] = []
        answers: List[str] = []
        for item in _ensure_list(record.get("neighborhood_prompts")):
            if not isinstance(item, dict):
                continue
            prompt = _text(item.get("prompt") or item.get("src") or item.get("text"))
            answer = _text(
                item.get("target")
                or item.get("ground_truth")
                or item.get("answer")
                or item.get("target_true")
            )
            if prompt and answer:
                prompts.append(prompt)
                answers.append(answer)
            if len(prompts) >= max_locality:
                break
        return prompts, answers

    locality_prompts = (
        record.get("locality_prompts")
        or record.get("locality_prompt")
        or record.get("loc")
    )
    locality_answers = (
        record.get("locality_answers")
        or record.get("locality_ground_truth")
        or record.get("locality_answer")
        or record.get("loc_ans")
    )
    return _pair_lists(_ensure_list(locality_prompts), _ensure_list(locality_answers), max_locality)


def _extract_portability_prompts(record: dict[str, Any], limit: int) -> List[str]:
    portability = record.get("portability")
    if not isinstance(portability, dict):
        return []

    prompts: List[str] = []
    for bucket in portability.values():
        for item in _ensure_list(bucket):
            if not isinstance(item, dict):
                continue
            prompt = _text(item.get("prompt"))
            if prompt:
                prompts.append(prompt)
            if len(prompts) >= limit:
                return _dedupe_nonempty(prompts, limit=limit)
    return _dedupe_nonempty(prompts, limit=limit)


def _extract_portability_answers(record: dict[str, Any], prompt_count: int) -> List[str]:
    answers = _ensure_list(
        record.get("portability_answers")
        or record.get("portability_ground_truth")
        or record.get("new_answers")
        or record.get("new_answer")
    )
    answer_texts = [_text(value) for value in answers if _text(value)]
    if not answer_texts:
        return []
    if len(answer_texts) == 1 and prompt_count > 1:
        return answer_texts * prompt_count
    return answer_texts[:prompt_count]


def _convert_counterfact_record(
    record: dict[str, Any],
    *,
    index: int,
    max_paraphrases: int,
    max_locality: int,
) -> dict[str, Any] | None:
    requested = record.get("requested_rewrite")
    if isinstance(requested, list):
        requested = requested[0] if requested else None
    requested = requested if isinstance(requested, dict) else {}

    subject = _text(
        requested.get("subject")
        or record.get("subject")
        or record.get("concept")
        or record.get("entity")
    )
    prompt = _text(requested.get("prompt") or record.get("prompt") or record.get("src"))
    if prompt:
        prompt = _format_prompt(prompt, subject)
    target = _text(requested.get("target_new") or record.get("target_new") or record.get("target"))
    ground_truth = _text(
        requested.get("target_true")
        or record.get("ground_truth")
        or record.get("target_true")
    )
    paraphrases = _dedupe_nonempty(
        _ensure_list(
            record.get("paraphrase_prompts")
            or record.get("paraphrases")
            or record.get("rephrase_prompt")
            or record.get("rephrase")
        ),
        limit=max_paraphrases,
    )
    if len(paraphrases) < max_paraphrases:
        portability_prompts = _extract_portability_prompts(
            record, limit=max_paraphrases - len(paraphrases)
        )
        paraphrases = _dedupe_nonempty(
            [*paraphrases, *portability_prompts],
            limit=max_paraphrases,
        )
    locality_prompts, locality_answers = _extract_locality_pairs(record, max_locality=max_locality)

    if not prompt or not target:
        return None
    if not subject:
        subject = prompt

    return {
        "id": str(record.get("case_id") or record.get("id") or f"counterfact:{index}"),
        "subject": subject,
        "relation": _text(requested.get("relation") or record.get("relation")) or "counterfact",
        "target": target,
        "prompt": prompt,
        "ground_truth": ground_truth,
        "paraphrases": paraphrases,
        "locality_prompts": locality_prompts,
        "locality_answers": locality_answers,
    }


def _convert_zsre_record(
    record: dict[str, Any],
    *,
    index: int,
    max_paraphrases: int,
    max_locality: int,
) -> dict[str, Any] | None:
    prompt = _text(record.get("src") or record.get("prompt"))
    target = _text(record.get("alt") or record.get("target_new") or record.get("target"))
    ground_truth = _text(record.get("answers") or record.get("ground_truth"))
    paraphrases = _dedupe_nonempty(
        _ensure_list(record.get("paraphrases") or record.get("rephrase") or record.get("rephrase_prompt")),
        limit=max_paraphrases,
    )
    if len(paraphrases) < max_paraphrases:
        portability_prompts = _extract_portability_prompts(
            record, limit=max_paraphrases - len(paraphrases)
        )
        paraphrases = _dedupe_nonempty(
            [*paraphrases, *portability_prompts],
            limit=max_paraphrases,
        )
    locality_prompts, locality_answers = _extract_locality_pairs(record, max_locality=max_locality)

    if not prompt or not target:
        return None

    return {
        "id": str(record.get("case_id") or record.get("id") or f"zsre:{index}"),
        "subject": _text(record.get("subject")) or prompt,
        "relation": _text(record.get("relation")) or "zsre",
        "target": target,
        "prompt": prompt,
        "ground_truth": ground_truth,
        "paraphrases": paraphrases,
        "locality_prompts": locality_prompts,
        "locality_answers": locality_answers,
    }


def _convert_mquake_records(
    record: dict[str, Any],
    *,
    index: int,
    max_paraphrases: int,
    max_locality: int,
    max_portability: int,
) -> List[dict[str, Any]]:
    requested_rewrites = [
        item for item in _ensure_list(record.get("requested_rewrite")) if isinstance(item, dict)
    ]
    if not requested_rewrites:
        return []

    group_id = str(record.get("case_id") or record.get("id") or f"mquake:{index}")
    locality_prompts, locality_answers = _extract_locality_pairs(
        record, max_locality=max_locality
    )
    portability_prompts = _dedupe_nonempty(
        _ensure_list(record.get("questions") or record.get("portability_prompt")),
        limit=max_portability,
    )
    portability_answers = _extract_portability_answers(record, len(portability_prompts))
    if portability_prompts and portability_answers and len(portability_prompts) != len(portability_answers):
        count = min(len(portability_prompts), len(portability_answers))
        portability_prompts = portability_prompts[:count]
        portability_answers = portability_answers[:count]

    converted: List[dict[str, Any]] = []
    for rewrite_index, rewrite in enumerate(requested_rewrites):
        subject = _text(rewrite.get("subject") or record.get("subject"))
        prompt = _text(rewrite.get("prompt") or record.get("prompt"))
        if prompt:
            prompt = _format_prompt(prompt, subject)
        target = _text(rewrite.get("target_new") or rewrite.get("target") or record.get("target"))
        if not prompt or not target:
            continue

        paraphrases = _dedupe_nonempty(
            [rewrite.get("question"), *(_ensure_list(record.get("paraphrases")))],
            limit=max_paraphrases,
        )
        if not subject:
            subject = prompt

        converted.append(
            {
                "id": f"{group_id}:{rewrite_index}",
                "group_id": group_id,
                "subject": subject,
                "relation": _text(rewrite.get("relation") or record.get("relation")) or "mquake",
                "target": target,
                "prompt": prompt,
                "ground_truth": _text(rewrite.get("target_true") or record.get("answer") or record.get("ground_truth")),
                "paraphrases": paraphrases,
                "locality_prompts": locality_prompts,
                "locality_answers": locality_answers,
                "portability_prompts": portability_prompts,
                "portability_answers": portability_answers,
            }
        )

    return converted


def _detect_format(records: Sequence[dict[str, Any]]) -> str:
    if not records:
        raise ValueError("Cannot infer format from an empty dataset.")
    sample = records[0]
    if (
        isinstance(sample.get("requested_rewrite"), list)
        and ("questions" in sample or "new_answer" in sample)
    ):
        return "mquake"
    if "requested_rewrite" in sample or "paraphrase_prompts" in sample:
        return "counterfact"
    if "src" in sample and ("alt" in sample or "answers" in sample or "loc" in sample):
        return "zsre"
    if "prompt" in sample and "target_new" in sample and ("loc" in sample or "loc_ans" in sample):
        return "zsre"
    if "prompt" in sample and ("target_new" in sample or "target" in sample):
        return "counterfact"
    raise ValueError("Could not auto-detect dataset format.")


def main() -> None:
    args = parse_args()
    records = _load_records(args.input)
    dataset_format = args.format if args.format != "auto" else _detect_format(records)

    if dataset_format == "mquake":
        converted = []
        for index, record in enumerate(records):
            converted.extend(
                _convert_mquake_records(
                    record,
                    index=index,
                    max_paraphrases=args.max_paraphrases,
                    max_locality=args.max_locality,
                    max_portability=args.max_portability,
                )
            )
    else:
        converter = _convert_counterfact_record if dataset_format == "counterfact" else _convert_zsre_record
        converted = [
            item
            for index, record in enumerate(records)
            if (
                item := converter(
                    record,
                    index=index,
                    max_paraphrases=args.max_paraphrases,
                    max_locality=args.max_locality,
                )
            )
            is not None
        ]

    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(converted)

    if args.offset:
        converted = converted[args.offset :]
    if args.limit is not None:
        converted = converted[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for item in converted:
            handle.write(json.dumps(item, ensure_ascii=True) + "\n")

    summary = {
        "input_path": str(args.input),
        "output_path": str(args.output),
        "format": dataset_format,
        "written_records": len(converted),
        "shuffle": args.shuffle,
        "seed": args.seed,
        "offset": args.offset,
        "limit": args.limit,
        "max_paraphrases": args.max_paraphrases,
        "max_locality": args.max_locality,
        "max_portability": args.max_portability,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
