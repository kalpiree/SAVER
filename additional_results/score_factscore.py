from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path


MODEL_NAMES = (
    "retrieval+ChatGPT", "retrieval+llama", "retrieval+llama+npm",
    "retrieval+ChatGPT+npm", "npm",
)
DEFAULT_DIRECTORY = Path.home() / ".cache" / "factscore"


def _read_generations(data):
    rows, seen = [], set()
    for line_number, line in enumerate(data.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Line {line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Line {line_number}: record must be an object")
        identifier = row.get("id")
        if isinstance(identifier, bool) or not isinstance(identifier, (str, int)) or not str(identifier).strip():
            raise ValueError(f"Line {line_number}: id must be a nonempty string or integer")
        identifier = str(identifier)
        if identifier in seen:
            raise ValueError(f"Line {line_number}: duplicate id {identifier!r}")
        seen.add(identifier)
        for key in ("topic", "prompt", "generation"):
            if not isinstance(row.get(key), str):
                raise ValueError(f"Line {line_number}: {key} must be a string")
            if key != "generation" and not row[key].strip():
                raise ValueError(f"Line {line_number}: {key} must not be empty")
        rows.append({key: row[key] for key in ("topic", "prompt", "generation")} | {"id": identifier})
    if not rows:
        raise ValueError("Input contains no generated records")
    return rows


def load_generations(path: str | Path) -> list[dict]:
    return _read_generations(Path(path).expanduser().read_bytes())


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def build_parser():
    parser = argparse.ArgumentParser(description="Score generated biography JSONL using the official FactScorer API.")
    parser.add_argument("--input", "--input-path", "--input_path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New JSON report path; never overwritten")
    parser.add_argument("--input-metadata", type=Path, help="Generation provenance JSON (default: INPUT.metadata.json, if present)")
    parser.add_argument("--model-name", "--model_name", choices=MODEL_NAMES, default="retrieval+ChatGPT")
    parser.add_argument("--knowledge-source", "--knowledge_source", default="enwiki-20230401")
    parser.add_argument("--data-dir", "--data_dir", type=Path, default=DEFAULT_DIRECTORY)
    parser.add_argument("--model-dir", "--model_dir", type=Path, default=DEFAULT_DIRECTORY)
    parser.add_argument("--cache-dir", "--cache_dir", type=Path, default=DEFAULT_DIRECTORY)
    parser.add_argument("--api-key-file", type=Path, help="API key file path; defaults to FACTSCORE_KEY_FILE")
    parser.add_argument("--gamma", type=int, default=10, help="Length penalty threshold; 0 disables it (default: 10)")
    parser.add_argument("--abstain-detection", "--abstain_detection_type", choices=("none", "generic", "perplexity_ai"), default="none")
    parser.add_argument("--cost-estimate", "--cost_estimate", choices=("consider_cache", "ignore_cache"), default="consider_cache")
    parser.add_argument("--batch-size", type=int, default=256, help="Retrieval batch size (default: 256)")
    parser.add_argument("--verbose", action="store_true")
    return parser


def score(args):
    if args.gamma < 0 or args.batch_size < 1:
        raise ValueError("gamma must be nonnegative and batch-size must be positive")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.knowledge_source):
        raise ValueError("knowledge-source must be a nonempty filename-safe name")
    input_path = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise ValueError("Output path already exists")
    data = input_path.read_bytes()
    rows = _read_generations(data)
    metadata_path = args.input_metadata
    if metadata_path is None:
        candidate = Path(str(input_path) + ".metadata.json")
        metadata_path = candidate if candidate.is_file() else None
    upstream = None
    if metadata_path is not None:
        metadata_path = metadata_path.expanduser().resolve()
        raw_metadata = metadata_path.read_bytes()
        upstream = {
            "path": str(metadata_path),
            "sha256": hashlib.sha256(raw_metadata).hexdigest(),
            "content": json.loads(raw_metadata.decode("utf-8-sig")),
        }
    key_file = args.api_key_file or os.environ.get("FACTSCORE_KEY_FILE")
    if not key_file:
        raise ValueError("Provide --api-key-file or set FACTSCORE_KEY_FILE to an API key file path")
    key_file = Path(key_file).expanduser().resolve()
    if not key_file.is_file() or key_file.stat().st_size == 0:
        raise ValueError("API key file must exist and be nonempty")
    directories = {name: str(getattr(args, name).expanduser().resolve()) for name in ("data_dir", "model_dir", "cache_dir")}
    data_dir = Path(directories["data_dir"])
    if not (data_dir / "demos").is_dir():
        raise ValueError(f"Atomic-fact demonstrations are missing: {data_dir / 'demos'}")
    db_path = data_dir / f"{args.knowledge_source}.db"
    knowledge_path = db_path if db_path.is_file() else data_dir / f"{args.knowledge_source}.jsonl"
    if not knowledge_path.is_file():
        raise ValueError(f"Knowledge source requires {db_path.name} or {args.knowledge_source}.jsonl in data-dir")
    if "llama" in args.model_name and not (Path(directories["model_dir"]) / "inst-llama-7B").is_dir():
        raise ValueError("model-dir must contain inst-llama-7B for a llama estimator")
    if Path(directories["cache_dir"]).exists() and not Path(directories["cache_dir"]).is_dir():
        raise ValueError("cache-dir must be a directory")
    settings = {
        "model_name": args.model_name,
        "knowledge_source": args.knowledge_source,
        **directories,
        "api_key_file": str(key_file),
        "gamma": args.gamma,
        "abstain_detection_type": None if args.abstain_detection == "none" else args.abstain_detection,
        "cost_estimate": args.cost_estimate,
        "batch_size": args.batch_size,
        "verbose": args.verbose,
    }
    try:
        module = importlib.import_module("factscore.factscorer")
    except ImportError as exc:
        raise RuntimeError("Install factscore and its dependencies in the scoring environment") from exc
    try:
        version = importlib.metadata.version("factscore")
    except importlib.metadata.PackageNotFoundError:
        version = None
    provenance = {
        "input_path": str(input_path),
        "input_sha256": hashlib.sha256(data).hexdigest(),
        "input_metadata": upstream,
        "row_count": len(rows),
        "settings": settings,
        "factscore_version": version,
        "factscorer_sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
        "knowledge_source_file": {
            "path": str(knowledge_path),
            "size_bytes": knowledge_path.stat().st_size,
            "mtime_ns": knowledge_path.stat().st_mtime_ns,
        },
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    scorer = module.FactScorer(
        model_name=args.model_name,
        **directories,
        openai_key=str(key_file),
        cost_estimate=args.cost_estimate,
        abstain_detection_type=settings["abstain_detection_type"],
        batch_size=args.batch_size,
    )
    result = scorer.get_score(
        topics=[row["topic"] for row in rows],
        generations=[row["generation"] for row in rows],
        gamma=args.gamma,
        knowledge_source=args.knowledge_source,
        verbose=args.verbose,
    )
    decisions = result.get("decisions")
    if not isinstance(decisions, list) or len(decisions) != len(rows):
        raise RuntimeError("FactScorer returned decisions not aligned with the input records")
    report = _json_safe({
        "schema_version": 1,
        "provenance": {**provenance, "finished_at": datetime.now(timezone.utc).isoformat()},
        "metrics": {key: value for key, value in result.items() if key != "decisions"},
        "records": [{**row, "decisions": decision} for row, decision in zip(rows, decisions)],
    })
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(payload)
    return report


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        score(args)
    except (OSError, ValueError, RuntimeError, TypeError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
