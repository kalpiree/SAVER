"""First-token risk evaluator for causal language models."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

from saver.eval.base import BaseRiskEvaluator
from saver.types import EditorProposal, EvaluationResult, ProbeBundle


class FirstTokenCausalLMEvaluator(BaseRiskEvaluator):
    """Score miscoverage using the first generated token only.

    This is the safest first implementation for SAVER because it avoids the
    complexity of multi-token decoding while still matching standard editing
    practice for factual cloze prompts.
    """

    def __init__(self, max_prompt_tokens: int = 256) -> None:
        self.max_prompt_tokens = max_prompt_tokens
        self._torch = None

    def _lazy_torch(self):
        if self._torch is None:
            import torch

            self._torch = torch
        return self._torch

    def _model_device(self, model: object):
        torch = self._lazy_torch()
        return next(model.parameters()).device

    def _first_target_id(self, tokenizer: object, target_text: str) -> int:
        prefixed = tokenizer(" " + target_text, add_special_tokens=False).input_ids
        if prefixed:
            return int(prefixed[0])

        plain = tokenizer(target_text, add_special_tokens=False).input_ids
        if plain:
            return int(plain[0])
        raise ValueError(f"Could not tokenize target text '{target_text}'.")

    def _next_token_miscoverage(
        self,
        model: object,
        tokenizer: object,
        prompt: str,
        target_text: str,
        beta_grid: Sequence[float],
    ) -> Dict[float, float]:
        torch = self._lazy_torch()
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_prompt_tokens,
        )
        device = self._model_device(model)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.inference_mode():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]
            probabilities = torch.softmax(logits, dim=-1)

        sorted_probabilities, sorted_ids = torch.sort(probabilities, descending=True)
        cumulative = torch.cumsum(sorted_probabilities, dim=0)
        target_id = self._first_target_id(tokenizer, target_text)
        target_rank = int((sorted_ids == target_id).nonzero(as_tuple=False)[0].item())

        results: Dict[float, float] = {}
        for beta in beta_grid:
            cutoff_index = int(torch.searchsorted(cumulative, torch.tensor(beta, device=cumulative.device)).item())
            results[float(beta)] = 0.0 if target_rank <= cutoff_index else 1.0
        return results

    def _average(self, values: Iterable[float]) -> float:
        values = list(values)
        if not values:
            return 0.0
        return sum(values) / len(values)

    def evaluate(
        self,
        proposal: EditorProposal,
        probe_bundle: ProbeBundle,
        beta_grid: Sequence[float],
        locality_weight: float,
    ) -> EvaluationResult:
        handle = proposal.handle
        model = getattr(handle, "edited_model")
        tokenizer = getattr(handle, "tokenizer")
        model.eval()

        locality_buckets: Dict[float, List[float]] = {float(beta): [] for beta in beta_grid}
        generality_buckets: Dict[float, List[float]] = {float(beta): [] for beta in beta_grid}

        for prompt, target in zip(probe_bundle.generality.prompts, probe_bundle.generality.targets):
            losses = self._next_token_miscoverage(model, tokenizer, prompt, target, beta_grid)
            for beta, loss in losses.items():
                generality_buckets[beta].append(loss)

        for prompt, target in zip(probe_bundle.locality.prompts, probe_bundle.locality.targets):
            losses = self._next_token_miscoverage(model, tokenizer, prompt, target, beta_grid)
            for beta, loss in losses.items():
                locality_buckets[beta].append(loss)

        locality_risk: Dict[float, float] = {}
        generality_risk: Dict[float, float] = {}
        joint: Dict[float, float] = {}
        for beta in beta_grid:
            beta = float(beta)
            locality_score = self._average(locality_buckets[beta])
            generality_score = self._average(generality_buckets[beta])
            locality_risk[beta] = locality_score
            generality_risk[beta] = generality_score
            joint[beta] = locality_weight * locality_score + (1.0 - locality_weight) * generality_score

        return EvaluationResult(
            generality_risk=generality_risk,
            locality_risk=locality_risk,
            joint_risk=joint,
            notes={"unit": "first_token"},
        )
