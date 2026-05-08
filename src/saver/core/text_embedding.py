"""Lightweight text embeddings used before sentence encoders are integrated."""

from __future__ import annotations

import hashlib
import re
from typing import List


def hashed_text_embedding(text: str, dimension: int) -> List[float]:
    """Deterministic token-aware embedding for early-stage structural-tension experiments.

    Averaging token hashes makes prompts with overlapping phrasing sit closer
    together than a raw whole-string hash. That is much more suitable for the
    current SAVER proxy, where repeated edits over similar relations should
    produce increasing structural tension.
    """

    tokens = re.findall(r"[a-z0-9]+", text.lower())
    if not tokens:
        tokens = [text.lower()]

    values = [0.0] * dimension
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        for index in range(dimension):
            byte_value = digest[index % len(digest)]
            values[index] += (byte_value / 127.5) - 1.0

    scale = 1.0 / len(tokens)
    return [value * scale for value in values]
