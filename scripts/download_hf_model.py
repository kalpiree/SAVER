#!/usr/bin/env python3
"""Download a Hugging Face model snapshot into a fixed local directory."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Hugging Face repo id, for example openai-community/gpt2-xl.",
    )
    parser.add_argument(
        "--local-dir",
        required=True,
        type=Path,
        help="Target directory for the downloaded snapshot.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional branch, tag, or commit hash.",
    )
    parser.add_argument(
        "--token-env",
        default="HF_TOKEN",
        help="Environment variable that stores the Hugging Face token.",
    )
    parser.add_argument(
        "--allow-pattern",
        action="append",
        default=[],
        help="Optional allow pattern. Can be passed multiple times.",
    )
    parser.add_argument(
        "--ignore-pattern",
        action="append",
        default=[],
        help="Optional ignore pattern. Can be passed multiple times.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = os.environ.get(args.token_env) or None
    local_dir = args.local_dir.expanduser().resolve()
    local_dir.mkdir(parents=True, exist_ok=True)

    downloaded_path = snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(local_dir),
        revision=args.revision,
        token=token,
        local_dir_use_symlinks=False,
        allow_patterns=args.allow_pattern or None,
        ignore_patterns=args.ignore_pattern or None,
    )

    print(
        json.dumps(
            {
                "repo_id": args.repo_id,
                "local_dir": str(local_dir),
                "downloaded_path": downloaded_path,
                "revision": args.revision,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
