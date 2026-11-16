from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from additional_results.runtime import (
    Controller, append_row, audit_scores, digest, freeze_audit, generate_biographies,
    load_config, make_adapter, read_rows, score_answer, write_json,
)
from saver.data.counterfact import load_counterfact_like_jsonl
from saver.eval.counterfact import CounterFactProbeGenerator
from saver.eval.metrics import freeze_locality_references


def validate_banks(edits, audit):
    if not edits or not audit:
        raise ValueError("Nonempty edit and audit banks are required.")
    if len({row["id"] for row in audit}) != len(audit) or len({row["prompt"] for row in audit}) != len(audit):
        raise ValueError("Audit ids and prompts must be unique.")
    heldout = {row["prompt"] for row in audit}
    online = {prompt for edit in edits for prompt in [
        edit.metadata["rewrite_prompt"], *edit.paraphrases, *edit.metadata.get("locality_prompts", []),
    ]}
    if heldout & online:
        raise ValueError("Held-out audit prompts overlap with editor or monitoring inputs.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "additional_results/configs/qwen35_alphaedit.json")
    parser.add_argument("--stream", type=Path, required=True)
    parser.add_argument("--audit-bank", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("saver", "unconstrained"), required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--biographies", type=Path)
    args = parser.parse_args()
    config, protocol = load_config(args.config, args.seed)
    limit = args.limit if args.limit is not None else int(protocol["limit"])
    edits = load_counterfact_like_jsonl(args.stream, require_validated=True)
    if limit < 2 or len(edits) < limit:
        raise ValueError("The edit stream must contain at least the requested number of edits.")
    edits = edits[:limit]
    if len({edit.metadata["source_id"] for edit in edits}) != len(edits):
        raise ValueError("Duplicate edit ids.")
    audit = read_rows(args.audit_bank)
    validate_banks(edits, audit)
    biographies = None
    if args.biographies:
        from additional_results.prepare_datasets import load_biographies
        biographies = load_biographies(args.biographies)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    adapter = make_adapter(config, args.mode)
    max_prompt = int(config.get("max_prompt_tokens", 256))
    references, excluded = freeze_audit(adapter.model, adapter.tokenizer, audit, protocol, max_prompt)
    reference_sha = hashlib.sha256(json.dumps(references, sort_keys=True).encode()).hexdigest()
    write_json(args.output_dir / "audit_references.json", {"references": references, "excluded_ids": excluded})
    online_references = freeze_locality_references(adapter.model, adapter.tokenizer, edits, max_prompt)
    controller = Controller(config, edits, online_references, args.mode)
    generator = CounterFactProbeGenerator()
    previous = audit_scores(adapter.model, adapter.tokenizer, references, protocol, max_prompt)
    initial = dict(previous)
    events = args.output_dir / "events.jsonl"
    summaries = []
    accepted_count = 0
    audit_seconds = 0.0
    checkpoints = set(protocol["checkpoints"]) | {limit}
    metadata = {
        "protocol_version": 1, "mode": args.mode, "seed": args.seed, "limit": limit,
        "completed_steps": 0, "config": config, "protocol": protocol,
        "events_path": str(events.resolve()), "stream_sha256": digest(args.stream),
        "audit_sha256": digest(args.audit_bank), "audit_reference_sha256": reference_sha,
        "audit_reference_count": len(references), "excluded_audit_ids": excluded,
        "initial_audit": initial, "checkpoints": summaries,
        "biographies_sha256": digest(args.biographies) if args.biographies else None,
    }
    write_json(args.output_dir / "run.json", metadata)
    for step, edit in enumerate(edits, 1):
        bundle = generator.build(edit)
        pre_ratio = controller.pre_ratio()
        proposal = adapter.propose(bundle)
        try:
            accepted, snapshot, post_ratio = controller.decide(proposal, bundle)
            audit_started = time.perf_counter()
            answer = score_answer(
                adapter.model, adapter.tokenizer, bundle.edit_prompt, edit.target, protocol, max_prompt,
            )
            candidate = audit_scores(adapter.model, adapter.tokenizer, references, protocol, max_prompt)
            audit_seconds += time.perf_counter() - audit_started
            benign = answer["correct"] and candidate["accuracy"] >= previous["accuracy"]
            committed = candidate if accepted else previous
            event = {
                "step": step, "source_id": edit.metadata["source_id"], "accepted": accepted,
                "pre_monitor_ratio": pre_ratio, "post_monitor_ratio": post_ratio,
                "benign": benign, "candidate_success": answer["correct"],
                "target_token_count": answer["token_count"], "first_token_generality_risk": answer["risk"],
                "full_answer_error": float(not answer["correct"]),
                "locality_first_token_risk": candidate["multi_risk"],
                "locality_complete_error": 1.0 - candidate["multi_accuracy"] if candidate["multi_accuracy"] is not None else None,
                "locality_before_accuracy": previous["multi_accuracy"],
                "locality_candidate_accuracy": candidate["multi_accuracy"],
                "locality_committed_accuracy": committed["multi_accuracy"],
                "multi_token_locality_count": candidate["multi_count"],
                "audit_before_accuracy": previous["accuracy"], "audit_candidate_accuracy": candidate["accuracy"],
                "sampled": snapshot.sampled if snapshot else None,
                "q_t": snapshot.q_t if snapshot else None,
                "chosen_beta": snapshot.chosen_beta if snapshot else None,
            }
        except BaseException:
            adapter.rollback(proposal)
            raise
        if accepted:
            adapter.commit(proposal)
            accepted_count += 1
            previous = candidate
        else:
            adapter.rollback(proposal)
        append_row(events, event)
        metadata["completed_steps"] = step
        if step in checkpoints:
            summaries.append({
                "step": step, "accepted": accepted_count, "acceptance_rate": accepted_count / step,
                "audit_accuracy": previous["accuracy"], "multi_token_audit_accuracy": previous["multi_accuracy"],
            })
            metadata["wall_seconds"] = time.perf_counter() - started
            metadata["diagnostic_seconds"] = audit_seconds
            write_json(args.output_dir / "run.json", metadata)
    if biographies:
        generate_biographies(adapter, biographies, args.output_dir / "biographies.jsonl", config, protocol)
    metadata["wall_seconds"] = time.perf_counter() - started
    write_json(args.output_dir / "run.json", metadata)


if __name__ == "__main__":
    main()
