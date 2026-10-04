from __future__ import annotations

import sys
import types


def _patch_transformers_pytorch_utils() -> None:
    try:
        import torch
        import transformers.pytorch_utils as pytorch_utils
    except Exception:
        return

    if hasattr(pytorch_utils, "find_pruneable_heads_and_indices"):
        return

    def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
        heads = set(heads) - set(already_pruned_heads)
        mask = torch.ones(n_heads, head_size)
        for head in heads:
            head = head - sum(1 if pruned_head < head else 0 for pruned_head in already_pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask), dtype=torch.long)[mask]
        return heads, index

    pytorch_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices


def _patch_transformers_generation_utils() -> None:
    try:
        import transformers.generation.utils as generation_utils
    except Exception:
        return

    def existing(*names):
        for name in names:
            value = getattr(generation_utils, name, None)
            if value is not None:
                return value
        return object

    aliases = {
        "GreedySearchOutput": existing("GenerateNonBeamOutput", "GenerateOutput"),
        "GreedySearchDecoderOnlyOutput": existing("GenerateDecoderOnlyOutput", "GenerateNonBeamOutput"),
        "GreedySearchEncoderDecoderOutput": existing(
            "GenerateEncoderDecoderOutput", "GenerateNonBeamOutput"
        ),
        "SampleDecoderOnlyOutput": existing("GenerateDecoderOnlyOutput", "GenerateNonBeamOutput"),
        "SampleEncoderDecoderOutput": existing(
            "GenerateEncoderDecoderOutput", "GenerateNonBeamOutput"
        ),
        "BeamSearchDecoderOnlyOutput": existing("GenerateBeamDecoderOnlyOutput", "GenerateOutput"),
        "BeamSearchEncoderDecoderOutput": existing(
            "GenerateBeamEncoderDecoderOutput", "GenerateOutput"
        ),
    }
    for name, value in aliases.items():
        if not hasattr(generation_utils, name):
            setattr(generation_utils, name, value)


def _patch_transformers_generation_beam_search() -> None:
    module_name = "transformers.generation.beam_search"
    if module_name in sys.modules:
        return

    beam_search = types.ModuleType(module_name)

    class BeamScorer:
        pass

    class BeamSearchScorer(BeamScorer):
        pass

    beam_search.BeamScorer = BeamScorer
    beam_search.BeamSearchScorer = BeamSearchScorer
    sys.modules[module_name] = beam_search


_patch_transformers_pytorch_utils()
_patch_transformers_generation_utils()
_patch_transformers_generation_beam_search()
