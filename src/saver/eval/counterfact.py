"""CounterFact-style probe generation for SAVER."""

from __future__ import annotations

from typing import List, Sequence

from saver.eval.base import BaseProbeGenerator
from saver.types import EditRequest, ProbeBundle, ProbeSet


class CounterFactProbeGenerator(BaseProbeGenerator):
    """Build rewrite, rephrase, portability, and locality probes from edit records."""

    def build(self, edit_request: EditRequest) -> ProbeBundle:
        metadata = edit_request.metadata
        rewrite_prompt = metadata.get("rewrite_prompt")
        if not isinstance(rewrite_prompt, str) or not rewrite_prompt:
            raise ValueError("CounterFact records must include a non-empty 'rewrite_prompt'.")

        generality_pairs = [(rewrite_prompt, edit_request.target)]
        for paraphrase in edit_request.paraphrases:
            pair = (paraphrase, edit_request.target)
            if pair not in generality_pairs:
                generality_pairs.append(pair)

        portability_prompts = metadata.get("portability_prompts", [])
        portability_answers = metadata.get("portability_answers", [])
        if len(portability_prompts) != len(portability_answers):
            raise ValueError("portability_prompts and portability_answers must have the same length.")
        for prompt, target in zip(portability_prompts, portability_answers):
            pair = (prompt, target)
            if pair not in generality_pairs:
                generality_pairs.append(pair)

        locality_prompts = metadata.get("locality_prompts", [])
        locality_answers = metadata.get("locality_answers", [])
        if len(locality_prompts) != len(locality_answers):
            raise ValueError("locality_prompts and locality_answers must have the same length.")

        return ProbeBundle(
            edit_request=edit_request,
            edit_prompt=rewrite_prompt,
            generality=ProbeSet(
                prompts=[prompt for prompt, _ in generality_pairs],
                targets=[target for _, target in generality_pairs],
            ),
            locality=ProbeSet(
                prompts=list(locality_prompts),
                targets=list(locality_answers),
            ),
        )
