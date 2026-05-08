#!/usr/bin/env python3
"""Validate that the current environment is ready for EasyEdit-backed SAVER runs."""

from __future__ import annotations

import argparse
import importlib
import json
import pathlib
import sys

try:
    from importlib.util import find_spec
except Exception:  # pragma: no cover - defensive fallback for odd Python envs
    find_spec = None

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from saver.runtime_config import apply_env_editor_overrides

LOCAL_EASYEDIT_ROOT = PROJECT_ROOT / "external" / "EasyEdit"
if LOCAL_EASYEDIT_ROOT.exists() and str(LOCAL_EASYEDIT_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_EASYEDIT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=PROJECT_ROOT / "configs" / "examples" / "rome_example.json",
        help="Path to the EasyEdit example config.",
    )
    return parser.parse_args()


def _module_available(name: str) -> bool:
    if find_spec is not None:
        try:
            return find_spec(name) is not None
        except Exception:
            return False

    try:
        __import__(name)
        return True
    except Exception:
        return False


def _module_import_diagnostic(name: str) -> dict:
    result = {
        "discoverable": _module_available(name),
        "importable": False,
        "error": None,
    }
    try:
        importlib.import_module(name)
        result["importable"] = True
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def _torch_cuda_status() -> dict:
    if not _module_available("torch"):
        return {
            "torch_version": None,
            "cuda_available": False,
            "cuda_device_count": 0,
            "cuda_devices": [],
        }

    try:
        import torch
    except Exception as exc:
        return {
            "torch_version": None,
            "cuda_available": False,
            "cuda_device_count": 0,
            "cuda_devices": [],
            "torch_import_error": str(exc),
        }

    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if cuda_available else 0
    device_names = []
    if cuda_available:
        for idx in range(device_count):
            try:
                device_names.append(torch.cuda.get_device_name(idx))
            except Exception:
                device_names.append(f"cuda:{idx}")

    return {
        "torch_version": getattr(torch, "__version__", None),
        "cuda_available": cuda_available,
        "cuda_device_count": device_count,
        "cuda_devices": device_names,
        "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
    }


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        config = apply_env_editor_overrides(json.load(handle))

    editor_cfg = config["editor"]
    hparams_path = PROJECT_ROOT / editor_cfg["hparams_path"]
    dataset_path = PROJECT_ROOT / config["dataset_path"]
    easyedit_checkout = PROJECT_ROOT / "external" / "EasyEdit" / "easyeditor"
    model_name_override = editor_cfg.get("model_name_override")
    model_override_exists = None
    if model_name_override and (
        str(model_name_override).startswith("/")
        or str(model_name_override).startswith("~")
        or str(model_name_override).startswith(".")
    ):
        model_override_exists = pathlib.Path(model_name_override).expanduser().exists()

    easyeditor_diag = _module_import_diagnostic("easyeditor")
    torch_diag = _module_import_diagnostic("torch")
    transformers_diag = _module_import_diagnostic("transformers")

    status = {
        "python_executable": sys.executable,
        "easyeditor_installed": easyeditor_diag["discoverable"],
        "easyeditor_importable": easyeditor_diag["importable"],
        "easyeditor_import_error": easyeditor_diag["error"],
        "easyeditor_checkout_exists": easyedit_checkout.exists(),
        "torch_installed": torch_diag["discoverable"],
        "torch_importable": torch_diag["importable"],
        "torch_import_error": torch_diag["error"],
        "transformers_installed": transformers_diag["discoverable"],
        "transformers_importable": transformers_diag["importable"],
        "transformers_import_error": transformers_diag["error"],
        "hparams_exists": hparams_path.exists(),
        "dataset_exists": dataset_path.exists(),
        "hparams_path": str(hparams_path),
        "dataset_path": str(dataset_path),
        "model_name_override": model_name_override,
        "model_name_override_exists": model_override_exists,
    }
    status.update(_torch_cuda_status())
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
