"""CounterFact-style data loading helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from saver.types import EditRequest


def load_counterfact_like_jsonl(path: str | Path) -> List[EditRequest]:
    """Load a light-weight CounterFact-style stream from JSONL.

    Expected fields per line:

    - `subject`
    - `relation`
    - `target`
    - `prompt`
    - `ground_truth` (optional)
    - `paraphrases` (optional)
    - `locality_prompts` (optional)
    - `locality_answers` (optional)
    - `portability_prompts` (optional)
    - `portability_answers` (optional)
    """

    source = Path(path)
    requests: List[EditRequest] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload: Dict[str, Any] = json.loads(line)
            try:
                prompt = payload["prompt"]
                requests.append(
                    EditRequest(
                        subject=payload["subject"],
                        relation=payload["relation"],
                        target=payload["target"],
                        paraphrases=tuple(payload.get("paraphrases", [])),
                        locality_subjects=tuple(payload.get("locality_subjects", [])),
                        metadata={
                            "rewrite_prompt": prompt,
                            "ground_truth": payload.get("ground_truth"),
                            "locality_prompts": payload.get("locality_prompts", []),
                            "locality_answers": payload.get("locality_answers", []),
                            "portability_prompts": payload.get("portability_prompts", []),
                            "portability_answers": payload.get("portability_answers", []),
                            "group_id": payload.get("group_id"),
                            "source_id": payload.get("id", f"{source.stem}:{line_number}"),
                        },
                    )
                )
            except KeyError as exc:
                raise KeyError(
                    f"Missing required field {exc!s} in {source} at line {line_number}."
                ) from exc
    return requests
