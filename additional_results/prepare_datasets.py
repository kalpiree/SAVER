from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import string
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, urlparse
from urllib.request import urlopen


MQUAKE_URL = (
    "https://huggingface.co/datasets/henryzhongsc/MQuAKE-Remastered/resolve/"
    "{revision}/data/CF3k-00000-of-00001.parquet"
)
BIOGRAPHY_PROMPT = "Question: Tell me a bio of {topic}."
MAX_SOURCE_BYTES = 128 * 1024 * 1024


def _text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _strings(value, field, required=False):
    if value is None and not required:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a string or list of strings")
    result = [_text(item, field) for item in value]
    if required and not result:
        raise ValueError(f"{field} must not be empty")
    return result


def _identifier(value):
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError("id must be a nonempty string or integer")
    return _text(str(value), "id")


def _unique(rows):
    seen = set()
    for row in rows:
        key = row["id"]
        if key in seen:
            raise ValueError(f"Duplicate id: {key!r}")
        seen.add(key)
    if not rows:
        raise ValueError("Input contains no records")
    return rows


def _parse(data, name, fmt="auto"):
    if fmt == "auto":
        suffix = Path(urlparse(str(name)).path).suffix.lower()
        fmt = {".txt": "txt", ".jsonl": "jsonl", ".ndjson": "jsonl"}.get(suffix)
        if fmt is None:
            fmt = "parquet" if data[:4] == b"PAR1" else "json"
    if fmt == "parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("Parquet input requires pyarrow") from exc
        rows = pq.read_table(io.BytesIO(data)).to_pylist()
    else:
        text = data.decode("utf-8-sig")
        if fmt == "txt":
            rows = [line.strip() for line in text.splitlines() if line.strip()]
        elif fmt == "jsonl":
            rows = []
            for line_number, line in enumerate(text.splitlines(), 1):
                if line.strip():
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid JSON on line {line_number}: {exc.msg}") from exc
        else:
            rows = json.loads(text)
            if isinstance(rows, dict):
                rows = [rows]
    if not isinstance(rows, list):
        raise ValueError("Input must contain a JSON array, JSONL records, topics TXT, or Parquet rows")
    return rows


def _target(edit, name):
    value = edit.get(name + "_str")
    if value is None:
        value = edit.get(name)
    if isinstance(value, dict):
        value = value.get("str")
    return _text(value, name)


def _rewrite_prompt(value, subject):
    parts = []
    for literal, field, specification, conversion in string.Formatter().parse(_text(value, "prompt")):
        parts.append(literal)
        if field is not None:
            if field not in ("", "0", "subject") or specification or conversion:
                raise ValueError("Rewrite prompt supports only {}, {0}, or {subject} fields")
            parts.append(subject)
    return "".join(parts)


def _mquake(rows):
    result = []
    for index, row in enumerate(rows, 1):
        try:
            if not isinstance(row, dict):
                raise ValueError("MQuAKE records must be objects")
            normalized = "rewrites" in row
            source_edits = row.get("rewrites" if normalized else "requested_rewrite")
            if not isinstance(source_edits, list) or not source_edits:
                raise ValueError("rewrites/requested_rewrite must be a nonempty list")
            rewrites = []
            for edit in source_edits:
                if not isinstance(edit, dict):
                    raise ValueError("Each rewrite must be an object")
                ground_truth = edit.get("ground_truth") if normalized else _target(edit, "target_true")
                if ground_truth is not None:
                    ground_truth = _text(ground_truth, "ground_truth")
                subject = _text(edit.get("subject"), "subject")
                rewrites.append({
                    "subject": subject,
                    "relation": _text(edit.get("relation", edit.get("relation_id")), "relation"),
                    "prompt": _rewrite_prompt(edit.get("prompt"), subject),
                    "target": _text(edit.get("target"), "target") if normalized else _target(edit, "target_new"),
                    "ground_truth": ground_truth,
                    "paraphrases": _strings(edit.get("paraphrases", []), "paraphrases"),
                })
            answers = row.get("answers") if normalized else row.get("new_answer", row.get("new_answers"))
            answers = _strings(answers, "new answers", required=True)
            answers += _strings(row.get("new_answer_alias", []), "new_answer_alias")
            record = dict(row)
            record.update(
                id=_identifier(row.get("id", row.get("case_id"))),
                questions=_strings(row.get("questions"), "questions", required=True),
                answers=list(dict.fromkeys(answers)),
                rewrites=rewrites,
            )
            result.append(record)
        except ValueError as exc:
            raise ValueError(f"MQuAKE record {index}: {exc}") from exc
    return _unique(result)


def _biographies(rows, prompt_template=BIOGRAPHY_PROMPT):
    fields = [field for _, field, _, _ in string.Formatter().parse(prompt_template) if field is not None]
    if not fields or any(field != "topic" for field in fields):
        raise ValueError("prompt-template must contain {topic} and no other replacement fields")
    result = []
    for index, raw in enumerate(rows, 1):
        try:
            if isinstance(raw, str):
                row = {"prompt": raw} if raw.startswith("Question: Tell me a bio of ") else {"topic": raw}
            elif isinstance(raw, dict):
                row = raw
            else:
                raise ValueError("Biography records must be topic strings or objects")
            topic = row.get("topic")
            if topic is None:
                match = re.fullmatch(r"Question: Tell me a bio of (.+)\.", row.get("prompt", ""))
                if match:
                    topic = match.group(1)
            topic = _text(topic, "topic").strip()
            prompt = row.get("prompt", prompt_template.format(topic=topic))
            result.append({
                "id": _identifier(row.get("id", f"biography-{index:06d}")),
                "topic": topic,
                "prompt": _text(prompt, "prompt"),
            })
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Biography record {index}: {exc}") from exc
    return _unique(result)


def load_mquake(path: str | Path) -> list[dict]:
    path = Path(path).expanduser()
    return _mquake(_parse(path.read_bytes(), path))


def load_biographies(path: str | Path) -> list[dict]:
    path = Path(path).expanduser()
    return _biographies(_parse(path.read_bytes(), path))


def _source(args):
    if args.input:
        path = args.input.expanduser().resolve()
        data = path.read_bytes()
        source = {"path": str(path), "version": args.source_version}
        name = str(path)
    else:
        url = args.url or MQUAKE_URL.format(revision=quote(args.revision, safe=""))
        if urlparse(url).scheme != "https":
            raise ValueError("Dataset URLs must use HTTPS")
        with urlopen(url, timeout=args.timeout) as response:
            if urlparse(response.url).scheme != "https":
                raise ValueError("Dataset redirect must use HTTPS")
            data = response.read(MAX_SOURCE_BYTES + 1)
            source = {
                "url": url,
                "resolved_url": response.url,
                "version": args.revision if args.download else args.source_version,
                "etag": response.headers.get("ETag"),
                "repo_commit": response.headers.get("X-Repo-Commit"),
            }
        name = url
    if len(data) > MAX_SOURCE_BYTES:
        raise ValueError(f"Source exceeds {MAX_SOURCE_BYTES} bytes")
    source["sha256"] = hashlib.sha256(data).hexdigest()
    if args.sha256 and source["sha256"] != args.sha256.lower():
        raise ValueError("Source SHA256 does not match --sha256")
    if args.archive_member:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            info = archive.getinfo(args.archive_member)
            if info.is_dir() or info.file_size > MAX_SOURCE_BYTES:
                raise ValueError("Archive member is not a supported size or is a directory")
            data = archive.read(info)
        name = args.archive_member
        source["archive_member"] = name
    source["payload_sha256"] = hashlib.sha256(data).hexdigest()
    return _parse(data, name, args.format), source


def build_parser():
    parser = argparse.ArgumentParser(description="Normalize MQuAKE groups or FActScore biography prompts to JSONL.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("mquake", "biographies"):
        command = commands.add_parser(name)
        source = command.add_mutually_exclusive_group(required=True)
        source.add_argument("--input", type=Path, help="Local JSON, JSONL, Parquet, or biography TXT/ZIP")
        source.add_argument("--url", help="Explicit HTTPS dataset URL")
        if name == "mquake":
            source.add_argument("--download", action="store_true", help="Download the official MQuAKE-Remastered CF3k Parquet")
            command.add_argument("--revision", default="main", help="Official HF revision for --download (default: main)")
        else:
            command.set_defaults(download=False)
            command.add_argument("--prompt-template", default=BIOGRAPHY_PROMPT, help="Template for topics without prompts; default: %(default)s")
        command.add_argument("--source-version", help="Version label for a local source or explicit URL")
        command.add_argument("--format", choices=("auto", "json", "jsonl", "parquet", "txt"), default="auto")
        command.add_argument("--archive-member", help="Exact member path inside a ZIP; no archive extraction")
        command.add_argument("--sha256", help="Expected SHA256 of source bytes before ZIP decoding")
        command.add_argument("--timeout", type=float, default=60.0, help="Download timeout in seconds (default: 60)")
        command.add_argument("--output", type=Path, required=True, help="New JSONL output path; never overwritten")
        command.add_argument("--metadata", type=Path, help="Provenance JSON path (default: OUTPUT.metadata.json)")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.timeout <= 0:
            raise ValueError("timeout must be positive")
        if args.sha256 and not re.fullmatch(r"[0-9a-fA-F]{64}", args.sha256):
            raise ValueError("sha256 must contain 64 hexadecimal characters")
        output = args.output.expanduser().resolve()
        metadata = args.metadata.expanduser().resolve() if args.metadata else Path(str(output) + ".metadata.json")
        if output == metadata or output.exists() or metadata.exists():
            raise ValueError("Output and metadata must be distinct, new paths")
        rows, source = _source(args)
        rows = _mquake(rows) if args.command == "mquake" else _biographies(rows, args.prompt_template)
        payload = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows)
        provenance = {
            "schema_version": 1,
            "dataset": args.command,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "row_count": len(rows),
            "output_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            "settings": {"format": args.format, "prompt_template": getattr(args, "prompt_template", None)},
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        metadata.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            handle.write(payload)
        with metadata.open("x", encoding="utf-8") as handle:
            json.dump(provenance, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    except (OSError, ValueError, RuntimeError, KeyError, zipfile.BadZipFile) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
