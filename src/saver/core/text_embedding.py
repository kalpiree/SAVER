from __future__ import annotations

import hashlib
import re
from typing import List, Sequence


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


class SentenceEmbedding:
    def __init__(self, reference_prompts: Sequence[str], config: dict) -> None:
        import numpy as np
        from transformers import AutoModel, AutoTokenizer

        prompts = list(dict.fromkeys(reference_prompts))
        if len(prompts) < 2:
            raise ValueError("Covariance estimation requires at least two reference prompts.")
        model_name = config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(config.get("embedding_device", "cpu"))
        self.model.eval()
        self.model.requires_grad_(False)
        self.batch_size = int(config.get("embedding_batch_size", 32))
        self.max_length = int(config.get("embedding_max_tokens", 256))
        epsilon = float(config.get("embedding_whitening_epsilon", 1e-5))
        if epsilon <= 0.0:
            raise ValueError("embedding_whitening_epsilon must be positive.")
        vectors = self._encode(prompts).astype(np.float64)
        self.mean = vectors.mean(axis=0)
        centered = vectors - self.mean
        covariance = centered.T @ centered / (len(prompts) - 1)
        values, axes = np.linalg.eigh(covariance)
        self.whitening = (axes * (1.0 / np.sqrt(np.maximum(values, 0.0) + epsilon))) @ axes.T
        corrected = centered @ self.whitening
        self.cache = {prompt: vector.tolist() for prompt, vector in zip(prompts, corrected)}

    def _encode(self, prompts: Sequence[str]):
        import numpy as np
        import torch

        vectors = []
        for start in range(0, len(prompts), self.batch_size):
            encoded = self.tokenizer(
                list(prompts[start:start + self.batch_size]), padding=True,
                truncation=True, max_length=self.max_length, return_tensors="pt",
            )
            encoded = {key: value.to(next(self.model.parameters()).device) for key, value in encoded.items()}
            with torch.inference_mode():
                hidden = self.model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
                pooled = torch.nn.functional.normalize(pooled, dim=-1)
            vectors.append(pooled.cpu().float().numpy())
        return np.concatenate(vectors, axis=0)

    def __call__(self, prompt: str) -> List[float]:
        if prompt not in self.cache:
            vector = self._encode([prompt])[0]
            self.cache[prompt] = ((vector - self.mean) @ self.whitening).tolist()
        return self.cache[prompt]
