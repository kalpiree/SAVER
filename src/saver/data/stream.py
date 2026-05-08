"""Helpers for loading sequential edit streams."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from saver.types import EditRequest


def load_edit_requests_jsonl(path: str | Path) -> List[EditRequest]:
    """Load a JSONL edit stream into typed edit requests."""

    source = Path(path)
    requests: List[EditRequest] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            try:
                requests.append(
                    EditRequest(
                        subject=payload["subject"],
                        relation=payload["relation"],
                        target=payload["target"],
                        paraphrases=tuple(payload.get("paraphrases", [])),
                        locality_subjects=tuple(payload.get("locality_subjects", [])),
                        metadata=payload.get("metadata", {}),
                    )
                )
            except KeyError as exc:
                raise KeyError(
                    f"Missing required field {exc!s} in {source} at line {line_number}."
                ) from exc
    return requests
