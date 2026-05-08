"""EasyEdit-backed editor adapter for SAVER."""

from __future__ import annotations

from copy import deepcopy
import importlib
import inspect
import pkgutil
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Callable, Dict, Mapping, Optional

from saver.editors.base import BaseEditorAdapter
from saver.types import EditorProposal, ProbeBundle


@dataclass
class EasyEditProposalHandle:
    """Metadata required to commit or roll back an EasyEdit candidate edit."""

    previous_model: object
    edited_model: object
    runtime_model: object
    tokenizer: object
    weights_copy: object
    request_payload: Mapping[str, object]
    editor_internal_state: object | None = None


def _import_required(module_name: str, install_hint: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        initial_error = exc
        project_root = Path(__file__).resolve().parents[3]
        local_easyedit = project_root / "external" / "EasyEdit"
        if module_name == "easyeditor" and local_easyedit.exists():
            sys.path.insert(0, str(local_easyedit))
            try:
                return importlib.import_module(module_name)
            except ImportError as local_exc:
                raise ImportError(
                    f"{module_name} import failed even with local EasyEdit checkout at "
                    f"{local_easyedit}. Original error: {initial_error}. "
                    f"Retry error: {local_exc}. {install_hint}"
                ) from local_exc
        raise ImportError(
            f"{module_name} import failed. Original error: {initial_error}. {install_hint}"
        ) from exc


class EasyEditAdapter(BaseEditorAdapter):
    """Adapter that uses EasyEdit's low-level apply function plus rollback weights."""

    def __init__(
        self,
        method: str,
        hparams_path: str | Path,
        overrides: Optional[Mapping[str, object]] = None,
    ) -> None:
        self.method = method.strip().lower()
        self.hparams_path = str(hparams_path)
        self.overrides = dict(overrides or {})
        self.easyeditor = _import_required(
            "easyeditor",
            "Install EasyEdit and its dependencies before running the real editor path.",
        )
        self.torch = _import_required(
            "torch",
            "Install PyTorch in the same environment as EasyEdit.",
        )

        self.hparams = self._load_hparams(self.hparams_path)
        self.editor = self._build_editor()
        self.model = self._extract_editor_attr("model")
        self.tokenizer = self._extract_editor_attr("tok", fallback="tokenizer")
        self.apply_algo = self._resolve_apply_function()

    def _method_token(self) -> str:
        return self.method.replace("-", "").replace("_", "")

    def _resolve_hparams_name(self) -> str:
        overrides = {
            "alphaedit": "AlphaEditHyperParams",
            "core": "COREHyperParams",
            "lora": "LoRAHyperParams",
            "memit": "MEMITHyperParams",
            "mend": "MENDHyperParams",
            "qlora": "QLoRAHyperParams",
            "serac": "SERACHparams",
            "ultraedit": "UltraEditHyperParams",
            "wise": "WISEHyperParams",
            "simie": "SimIEHyperParams",
            "rome": "ROMEHyperParams",
            "grace": "GraceHyperParams",
        }
        token = self._method_token()
        if token in overrides:
            return overrides[token]
        return f"{token.capitalize()}HyperParams"

    def _load_hparams(self, hparams_path: str) -> object:
        hparams_name = self._resolve_hparams_name()
        if not hasattr(self.easyeditor, hparams_name):
            raise AttributeError(
                f"EasyEdit does not expose {hparams_name}. "
                f"Check the method name '{self.method}' and the installed EasyEdit version."
            )
        hparams_cls = getattr(self.easyeditor, hparams_name)
        hparams = hparams_cls.from_hparams(hparams_path)
        self._apply_overrides(hparams)
        return hparams

    def _apply_overrides(self, hparams: object) -> None:
        for key, value in self.overrides.items():
            if value is None:
                continue
            if not hasattr(hparams, key):
                raise KeyError(f"Unsupported EasyEdit override '{key}'.")
            setattr(hparams, key, value)

    def _build_editor(self) -> object:
        if not hasattr(self.easyeditor, "BaseEditor"):
            raise AttributeError("EasyEdit does not expose BaseEditor in this environment.")
        base_editor = getattr(self.easyeditor, "BaseEditor")
        return base_editor.from_hparams(self.hparams)

    def _extract_editor_attr(self, primary: str, fallback: str | None = None) -> object:
        if hasattr(self.editor, primary):
            return getattr(self.editor, primary)
        if fallback and hasattr(self.editor, fallback):
            return getattr(self.editor, fallback)
        raise AttributeError(f"EasyEdit editor is missing attribute '{primary}'.")

    def _resolve_apply_function(self) -> Callable[..., object]:
        function_name = f"apply_{self._method_token()}_to_model"
        function_name_lower = function_name.lower()
        if hasattr(self.easyeditor, function_name):
            return getattr(self.easyeditor, function_name)
        for attr_name in dir(self.easyeditor):
            if attr_name.lower() == function_name_lower:
                return getattr(self.easyeditor, attr_name)

        models_module = _import_required(
            "easyeditor.models",
            "EasyEdit models package is missing from the installation.",
        )
        for module_info in pkgutil.walk_packages(
            models_module.__path__,
            prefix=f"{models_module.__name__}.",
        ):
            if self._method_token() not in module_info.name.replace("_", "").lower():
                continue
            try:
                module = importlib.import_module(module_info.name)
            except Exception:
                continue
            if hasattr(module, function_name):
                return getattr(module, function_name)
            for attr_name in dir(module):
                if attr_name.lower() == function_name_lower:
                    return getattr(module, attr_name)

        try:
            alg_dict_module = importlib.import_module("easyeditor.util.alg_dict")
            alg_dict = getattr(alg_dict_module, "ALG_DICT", {})
            alg_key = getattr(self.hparams, "alg_name", "").upper()
            if alg_key in alg_dict:
                return alg_dict[alg_key]
        except Exception:
            pass

        raise AttributeError(
            f"Could not find EasyEdit apply function '{function_name}'. "
            "The installed EasyEdit version may have changed its module layout."
        )

    def _sync_editor_state(self) -> None:
        if hasattr(self.editor, "model"):
            setattr(self.editor, "model", self.model)
        if hasattr(self.editor, "tok"):
            setattr(self.editor, "tok", self.tokenizer)
        if hasattr(self.editor, "tokenizer"):
            setattr(self.editor, "tokenizer", self.tokenizer)

    def _build_request_payload(self, probe_bundle: ProbeBundle) -> Dict[str, object]:
        metadata = probe_bundle.edit_request.metadata
        locality_prompt = None
        locality_target = None
        if probe_bundle.locality.prompts and probe_bundle.locality.targets:
            locality_prompt = probe_bundle.locality.prompts[0]
            locality_target = probe_bundle.locality.targets[0]
        loc_prompt = None
        if isinstance(locality_prompt, str) and locality_prompt:
            loc_prompt = locality_prompt
            if isinstance(locality_target, str) and locality_target:
                loc_prompt = f"{loc_prompt} {locality_target}"

        return {
            "prompt": probe_bundle.edit_prompt,
            "subject": probe_bundle.edit_request.subject,
            "target_new": probe_bundle.edit_request.target,
            "ground_truth": metadata.get("ground_truth"),
            "case_id": metadata.get("source_id", "saver-case"),
            "loc_prompt": loc_prompt,
            "rephrase_prompt": (
                probe_bundle.edit_request.paraphrases[0]
                if probe_bundle.edit_request.paraphrases
                else None
            ),
            "locality": (
                {
                    "neighborhood": {
                        "prompt": locality_prompt,
                        "ground_truth": locality_target,
                    }
                }
                if loc_prompt is not None
                else {}
            ),
            "portability": {},
        }

    def _resolve_runtime_model(self, edited_model: object) -> object:
        # WISE returns its own nn.Module wrapper around the edited HF model.
        # Keeping that wrapper as the long-lived runtime object breaks the next
        # edit step because EasyEdit inner-parameter paths like
        # `model.layers[29].mlp.down_proj.weight` get resolved against the
        # wrapper instead of the underlying causal LM.
        if self._method_token() == "wise" and hasattr(edited_model, "model"):
            return getattr(edited_model, "model")

        # Hugging Face causal LMs (and PEFT wrappers) are already the runtime
        # model we want to keep across sequential edits. Unwrapping their
        # `.model` attribute strips the top-level module prefix and breaks later
        # parameter lookups such as `model.layers.*` on the next edit step.
        if isinstance(edited_model, self.torch.nn.Module):
            return edited_model
        if hasattr(edited_model, "model"):
            return getattr(edited_model, "model")
        return edited_model

    def _snapshot_internal_state(self) -> object | None:
        if self._method_token() != "ultraedit":
            return None
        executor = getattr(self.apply_algo, "__self__", None)
        if executor is None or not hasattr(executor, "is_init"):
            return None
        if not executor.is_init:
            return {"is_init": False}
        return {
            "is_init": True,
            "model_state": deepcopy(executor.model.state_dict()),
            "alg_state": deepcopy(executor.alg.state_dict()),
        }

    def _restore_internal_state(self, state: object | None) -> None:
        if self._method_token() != "ultraedit" or state is None:
            return
        executor = getattr(self.apply_algo, "__self__", None)
        if executor is None or not hasattr(executor, "is_init"):
            return
        if not state.get("is_init"):
            if getattr(executor, "is_init", False):
                executor.reset_model()
            return
        executor.model.load_state_dict(state["model_state"])
        executor.alg.load_state_dict(state["alg_state"])
        executor.is_init = True

    def _restore_weights(self, model: object, weights_copy: Mapping[str, object]) -> None:
        named_parameters = dict(model.named_parameters())
        named_buffers = dict(model.named_buffers())
        with self.torch.no_grad():
            for name, original_value in weights_copy.items():
                if name in named_parameters:
                    target = named_parameters[name]
                elif name in named_buffers:
                    target = named_buffers[name]
                else:
                    raise KeyError(f"Unable to restore parameter '{name}'.")
                target.copy_(original_value.to(target.device))

    def propose(self, probe_bundle: ProbeBundle) -> EditorProposal:
        request_payload = self._build_request_payload(probe_bundle)
        previous_model = self.model
        editor_internal_state = self._snapshot_internal_state()
        apply_kwargs = {
            "model": self.model,
            "tok": self.tokenizer,
            "hparams": self.hparams,
            "copy": False,
            "return_orig_weights": True,
            "keep_original_weight": True,
        }
        apply_params = inspect.signature(self.apply_algo).parameters
        if "requests" in apply_params:
            apply_kwargs["requests"] = [request_payload]
        elif "request" in apply_params:
            apply_kwargs["request"] = [request_payload]
        else:
            apply_kwargs["requests"] = [request_payload]

        edited_model, weights_copy = self.apply_algo(**apply_kwargs)
        runtime_model = self._resolve_runtime_model(edited_model)
        self.model = runtime_model
        self._sync_editor_state()

        return EditorProposal(
            metadata={
                "method": self.method,
                "prompt": probe_bundle.edit_prompt,
                "target": probe_bundle.edit_request.target,
            },
            handle=EasyEditProposalHandle(
                previous_model=previous_model,
                edited_model=edited_model,
                runtime_model=runtime_model,
                tokenizer=self.tokenizer,
                weights_copy=weights_copy,
                request_payload=request_payload,
                editor_internal_state=editor_internal_state,
            ),
        )

    def commit(self, proposal: EditorProposal) -> None:
        handle = proposal.handle
        self.model = handle.runtime_model
        self.tokenizer = handle.tokenizer
        self._sync_editor_state()

    def rollback(self, proposal: EditorProposal) -> None:
        handle = proposal.handle
        if callable(handle.weights_copy):
            handle.weights_copy()
            self.model = handle.previous_model
        elif handle.previous_model is handle.runtime_model:
            self._restore_weights(handle.runtime_model, handle.weights_copy)
            self.model = handle.previous_model
        else:
            self.model = handle.previous_model
        self._restore_internal_state(handle.editor_internal_state)
        self.tokenizer = handle.tokenizer
        self._sync_editor_state()
