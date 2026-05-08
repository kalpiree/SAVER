#!/usr/bin/env python3
"""Generate controlled RQ4 stress streams from CounterFact-style JSONL data."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import re
from typing import Iterable


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
STOPWORDS = {
    "the",
    "a",
    "an",
    "of",
    "is",
    "are",
    "which",
    "what",
    "who",
    "where",
    "when",
    "for",
    "to",
    "in",
    "on",
    "and",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=pathlib.Path,
        default=pathlib.Path("data/streams/counterfact_full.jsonl"),
        help="Source CounterFact-style JSONL stream.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Number of edits per generated stress stream.",
    )
    parser.add_argument(
        "--pool",
        type=int,
        default=900,
        help="Candidate pool size used to mine overlaps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help="Random seed.",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("data/streams"),
        help="Output directory for generated JSONL files.",
    )
    return parser.parse_args()


def _load_jsonl(path: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: pathlib.Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _tokens(row: dict) -> set[str]:
    fields = [row.get("subject", ""), row.get("prompt", "")]
    fields.extend(row.get("paraphrases", []))
    fields.extend(row.get("locality_prompts", []))
    tokens: set[str] = set()
    for field in fields:
        for token in TOKEN_RE.findall(str(field).lower()):
            if token and token not in STOPWORDS:
                tokens.add(token)
    return tokens


def _score_overlap(tokens_a: set[str], tokens_b: set[str], relation_a: str, relation_b: str) -> float:
    if not tokens_a or not tokens_b:
        base = 0.0
    else:
        intersection = len(tokens_a & tokens_b)
        union = len(tokens_a | tokens_b)
        base = (intersection / union) if union else 0.0
    if relation_a == relation_b:
        base += 0.10
    return base


def _context_scores(rows: list[dict], pool_size: int) -> tuple[list[set[str]], list[list[tuple[int, float]]], list[float]]:
    pool = rows[: min(pool_size, len(rows))]
    token_sets = [_tokens(row) for row in pool]
    neighbor_lists: list[list[tuple[int, float]]] = [[] for _ in pool]
    hardness: list[float] = [0.0 for _ in pool]

    for i in range(len(pool)):
        for j in range(i + 1, len(pool)):
            score = _score_overlap(
                token_sets[i],
                token_sets[j],
                str(pool[i].get("relation", "")),
                str(pool[j].get("relation", "")),
            )
            if score <= 0.0:
                continue
            neighbor_lists[i].append((j, score))
            neighbor_lists[j].append((i, score))

    for idx, neighbors in enumerate(neighbor_lists):
        neighbors.sort(key=lambda item: item[1], reverse=True)
        top = [score for _neighbor, score in neighbors[:5]]
        hardness[idx] = (sum(top) / len(top)) if top else 0.0

    return token_sets, neighbor_lists, hardness


def _clone_row(row: dict, *, new_id: str, target: str | None = None, group_id: str | None = None) -> dict:
    clone = dict(row)
    clone["id"] = new_id
    if target is not None:
        clone["target"] = target
    if group_id is not None:
        clone["group_id"] = group_id
    return clone


def _build_collision_burst(rows: list[dict], neighbor_lists: list[list[tuple[int, float]]], limit: int) -> list[dict]:
    pool_size = len(neighbor_lists)
    unused = set(range(pool_size))
    stream: list[dict] = []
    burst_index = 0

    while unused and len(stream) < limit:
        anchor = max(
            unused,
            key=lambda idx: sum(score for neighbor, score in neighbor_lists[idx][:4] if neighbor in unused),
        )
        burst_members = [anchor]
        for neighbor, _score in neighbor_lists[anchor]:
            if neighbor in unused and neighbor != anchor:
                burst_members.append(neighbor)
            if len(burst_members) >= 5:
                break
        if len(burst_members) == 1:
            fallback = sorted(unused - {anchor})[:4]
            burst_members.extend(fallback)

        for member in burst_members:
            if member not in unused or len(stream) >= limit:
                continue
            unused.remove(member)
            row = _clone_row(
                rows[member],
                new_id=f"{rows[member].get('id', f'row:{member}')}:collision:{burst_index}",
                group_id=f"collision:{burst_index}",
            )
            stream.append(row)
        burst_index += 1

    if len(stream) < limit:
        for idx in range(pool_size):
            if len(stream) >= limit:
                break
            if idx in unused:
                unused.remove(idx)
                stream.append(
                    _clone_row(
                        rows[idx],
                        new_id=f"{rows[idx].get('id', f'row:{idx}')}:collisionfill",
                        group_id="collision:fill",
                    )
                )

    return stream[:limit]


def _pick_alternative_targets(rows: list[dict], start: int, count: int, banned: set[str]) -> list[str]:
    choices: list[str] = []
    cursor = start
    while cursor < len(rows) and len(choices) < count:
        candidate = str(rows[cursor].get("target", ""))
        if candidate and candidate not in banned:
            choices.append(candidate)
            banned.add(candidate)
        cursor += 1
    return choices


def _build_contradictory_loop(rows: list[dict], limit: int) -> list[dict]:
    anchors_needed = math.ceil(limit / 4)
    anchors = rows[:anchors_needed]
    stream: list[dict] = []
    donor_cursor = anchors_needed

    for anchor_index, anchor in enumerate(anchors):
        original_target = str(anchor.get("target", ""))
        banned = {original_target}
        alt_targets = _pick_alternative_targets(rows, donor_cursor, 2, banned)
        donor_cursor += max(1, len(alt_targets))
        if len(alt_targets) < 2:
            break
        sequence = [original_target, alt_targets[0], original_target, alt_targets[1]]
        group_id = f"contradict:{anchor_index}"
        for step_index, target in enumerate(sequence):
            stream.append(
                _clone_row(
                    anchor,
                    new_id=f"{anchor.get('id', f'anchor:{anchor_index}')}:contradict:{step_index}",
                    target=target,
                    group_id=group_id,
                )
            )
            if len(stream) >= limit:
                return stream

    return stream[:limit]


def _build_phase_shift(rows: list[dict], hardness: list[float], limit: int) -> list[dict]:
    candidate_count = min(len(hardness), len(rows))
    ranked = sorted(range(candidate_count), key=lambda idx: (hardness[idx], idx))
    half = limit // 2
    easy_ids = ranked[:half]
    hard_ids = list(reversed(ranked[-(limit - half) :]))
    selected: list[int] = []
    used: set[int] = set()

    for idx in easy_ids + hard_ids:
        if idx in used:
            continue
        used.add(idx)
        selected.append(idx)
        if len(selected) >= limit:
            break

    if len(selected) < limit:
        for idx in ranked:
            if idx in used:
                continue
            used.add(idx)
            selected.append(idx)
            if len(selected) >= limit:
                break

    stream: list[dict] = []
    for position, idx in enumerate(selected[:limit]):
        phase = "easy" if position < half else "hard"
        stream.append(
            _clone_row(
                rows[idx],
                new_id=f"{rows[idx].get('id', f'row:{idx}')}:phase:{position}",
                group_id=f"phase:{phase}",
            )
        )
    return stream


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    rows = _load_jsonl(args.source)
    rng.shuffle(rows)

    _token_sets, neighbor_lists, hardness = _context_scores(rows, args.pool)

    collision = _build_collision_burst(rows, neighbor_lists, args.limit)
    contradictory = _build_contradictory_loop(rows, args.limit)
    phase_shift = _build_phase_shift(rows, hardness, args.limit)

    outputs = {
        f"counterfact_rq4_collision_burst_{args.limit}.jsonl": collision,
        f"counterfact_rq4_contradictory_loop_{args.limit}.jsonl": contradictory,
        f"counterfact_rq4_phase_shift_{args.limit}.jsonl": phase_shift,
    }

    for name, stream in outputs.items():
        path = args.output_dir / name
        _write_jsonl(path, stream)
        print(f"WROTE {path} rows={len(stream)}")


if __name__ == "__main__":
    main()
