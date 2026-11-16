from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from additional_results.prepare_datasets import load_mquake
from additional_results.runtime import (
    Controller, append_row, digest, generate, load_config, make_adapter, normalize_answer, read_rows, write_json,
)
from saver.eval.counterfact import CounterFactProbeGenerator
from saver.eval.metrics import freeze_locality_references
from saver.types import EditRequest


def build_edits(cases, locality):
    groups = []
    for case in cases:
        group = []
        for index, rewrite in enumerate(case["rewrites"]):
            group.append(EditRequest(
                subject=rewrite["subject"], relation=rewrite["relation"], target=rewrite["target"],
                paraphrases=tuple(rewrite.get("paraphrases", [])),
                metadata={
                    "rewrite_prompt": rewrite["prompt"], "ground_truth": rewrite.get("ground_truth"),
                    "source_id": f"{case['id']}:{index}",
                    "locality_prompts": [row["prompt"] for row in locality],
                    "locality_answers": [row.get("target", "") for row in locality],
                },
            ))
        groups.append(group)
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "additional_results/configs/qwen35_alphaedit.json")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--locality-bank", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("saver", "unconstrained"), required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--locality-size", type=int, default=16)
    parser.add_argument("--case-mode", choices=("stream", "independent"))
    args = parser.parse_args()
    config, protocol = load_config(args.config, args.seed)
    case_mode = args.case_mode or protocol["mquake_case_mode"]
    if case_mode not in {"stream", "independent"}:
        raise ValueError("Invalid mquake_case_mode.")
    cases = load_mquake(args.dataset)
    if args.limit is not None:
        if args.limit < 1 or args.limit > len(cases):
            raise ValueError("Invalid case limit.")
        cases = cases[:args.limit]
    locality = read_rows(args.locality_bank)
    if args.locality_size < 1 or len(locality) < args.locality_size:
        raise ValueError("Insufficient locality probes.")
    locality = locality[:args.locality_size]
    online = {row["prompt"] for row in locality}
    questions = {q for case in cases for q in case["questions"]}
    edit_prompts = {r["prompt"] for case in cases for r in case["rewrites"]}
    formatted_questions = {protocol["question_template"].format(question=q) for q in questions}
    if len(online) != len(locality) or online & (questions | edit_prompts | formatted_questions):
        raise ValueError("Locality prompts must be unique and separate from edited facts and evaluation questions.")
    groups = build_edits(cases, locality)
    all_edits = [edit for group in groups for edit in group]
    if args.mode == "saver" and len({edit.metadata["rewrite_prompt"] for edit in all_edits}) < 2:
        raise ValueError("At least two unique edit prompts are required for the embedding covariance.")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    adapter = make_adapter(config, args.mode)
    references = freeze_locality_references(adapter.model, adapter.tokenizer, all_edits, config["max_prompt_tokens"])
    controller = Controller(config, all_edits, references, args.mode)
    generator = CounterFactProbeGenerator()
    committed = 0
    attempted = 0
    hits = 0
    question_count = 0
    case_scores = []
    case_successes = 0
    metadata = {
        "mode": args.mode, "seed": args.seed, "case_mode": case_mode,
        "config": config, "protocol": protocol, "dataset_sha256": digest(args.dataset),
        "locality_sha256": digest(args.locality_bank), "locality_size": args.locality_size,
        "case_count": len(cases), "completed_cases": 0,
    }
    for case_index, (case, edits) in enumerate(zip(cases, groups), 1):
        if case_mode == "independent":
            case_config = dict(config, seed=args.seed + case_index - 1)
            controller = Controller(case_config, all_edits, references, args.mode, embedding=controller.embedding)
        retained = []
        case_hits = 0
        try:
            for edit in edits:
                bundle = generator.build(edit)
                proposal = adapter.propose(bundle)
                try:
                    accepted, snapshot, ratio = controller.decide(proposal, bundle)
                except BaseException:
                    adapter.rollback(proposal)
                    raise
                attempted += 1
                if accepted:
                    adapter.commit(proposal)
                    committed += 1
                    if case_mode == "independent":
                        retained.append(proposal)
                else:
                    adapter.rollback(proposal)
                append_row(args.output_dir / "edits.jsonl", {
                    "case_id": case["id"], "source_id": edit.metadata["source_id"],
                    "step": attempted, "accepted": accepted, "post_monitor_ratio": ratio,
                    "chosen_beta": snapshot.chosen_beta if snapshot else None,
                })
            answers = {normalize_answer(answer) for answer in case["answers"]}
            for index, question in enumerate(case["questions"]):
                prompt = protocol["question_template"].format(question=question)
                result = generate(adapter.model, adapter.tokenizer, prompt, config["max_prompt_tokens"], protocol["mquake_max_new_tokens"])
                correct = result["completed"] and normalize_answer(result["text"]) in answers
                case_hits += int(correct)
                question_count += 1
                hits += int(correct)
                append_row(args.output_dir / "answers.jsonl", {
                    "case_id": case["id"], "question_index": index, "question": question,
                    "answers": case["answers"], "prediction": result["text"],
                    "completed": result["completed"], "exact_match": correct,
                })
        finally:
            for proposal in reversed(retained):
                adapter.rollback(proposal)
        case_scores.append(case_hits / len(case["questions"]))
        case_successes += int(case_hits > 0)
        metadata.update({
            "completed_cases": case_index, "attempted_edits": attempted, "accepted_edits": committed,
            "acceptance_rate": committed / attempted, "question_count": question_count,
            "exact_match": hits / question_count, "macro_case_exact_match": sum(case_scores) / len(case_scores),
            "case_exact_match": case_successes / case_index, "case_success_count": case_successes,
        })
        write_json(args.output_dir / "run.json", metadata)


if __name__ == "__main__":
    main()
