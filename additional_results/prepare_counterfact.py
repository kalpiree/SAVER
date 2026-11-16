from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from additional_results.runtime import append_row, digest, read_rows, write_json


def split_records(rows, audit_size, seed):
    if audit_size < 1 or not rows:
        raise ValueError("A positive audit size and a nonempty stream are required.")
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate edit ids.")
    generality = {prompt for row in rows for prompt in [row["prompt"], *row.get("paraphrases", [])]}
    owners = {}
    remaining = []
    for index, row in enumerate(rows):
        if row.get("probe_schema_version") != 2:
            raise ValueError("Rebuild the source stream with the current converter.")
        prompts = row.get("locality_prompts", [])
        answers = row.get("locality_answers", [])
        if len(prompts) != len(answers) or len(prompts) != len(set(prompts)):
            raise ValueError("Locality prompts must be unique and paired with answers.")
        if not prompts or set(prompts) & set([row["prompt"], *row.get("paraphrases", [])]):
            raise ValueError("Each edit needs nonempty locality probes disjoint from generality.")
        remaining.append(len(prompts))
        for prompt in prompts:
            owners.setdefault(prompt, []).append(index)
    candidates = sorted(set(owners) - generality)
    random.Random(seed).shuffle(candidates)
    heldout = []
    for prompt in candidates:
        if all(remaining[index] > 1 for index in owners[prompt]):
            heldout.append(prompt)
            for index in owners[prompt]:
                remaining[index] -= 1
        if len(heldout) == audit_size:
            break
    if len(heldout) != audit_size:
        raise ValueError("Insufficient disjoint locality prompts for the requested audit bank.")
    audit_set = set(heldout)
    stream = []
    for row in rows:
        updated = dict(row)
        pairs = [(p, a) for p, a in zip(row["locality_prompts"], row["locality_answers"]) if p not in audit_set]
        updated["locality_prompts"] = [p for p, _ in pairs]
        updated["locality_answers"] = [a for _, a in pairs]
        stream.append(updated)
    audit = [{"id": hashlib.sha256(prompt.encode()).hexdigest(), "prompt": prompt} for prompt in heldout]
    return stream, audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--audit-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    rows = read_rows(args.input)
    if args.limit < 1 or len(rows) < args.limit:
        raise ValueError("The source stream is shorter than the requested limit.")
    stream, audit = split_records(rows[:args.limit], args.audit_size, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, values in (("stream.jsonl", stream), ("audit.jsonl", audit)):
        for row in values:
            append_row(args.output_dir / name, row)
    write_json(args.output_dir / "manifest.json", {
        "source_sha256": digest(args.input), "seed": args.seed, "limit": args.limit,
        "audit_size": len(audit), "stream_sha256": digest(args.output_dir / "stream.jsonl"),
        "audit_sha256": digest(args.output_dir / "audit.jsonl"),
    })


if __name__ == "__main__":
    main()
