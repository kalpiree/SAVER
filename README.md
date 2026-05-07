# SAVER

This repository is the anonymous release scaffold for `SAVER`.
It is intentionally minimal and double-blind friendly: no author names,
affiliations, machine-specific paths, cluster identifiers, private logs, or
unpublished artifacts are included here by default.

## Repository Layout

- `src/`: core source code for the anonymous release
- `scripts/`: runnable experiment and utility scripts
- `configs/`: experiment configuration files
- `data/`: public metadata, toy examples, or download instructions
- `docs/`: reproduction notes and usage documentation
- `results/`: sanitized tables or summary outputs
- `external/`: third-party code pointers or wrappers that are safe to release
- `tests/`: lightweight checks and regression tests
- `assets/`: figures or static media for documentation

## Release Policy

Before publishing files into this repository, remove or sanitize:

- author names, emails, affiliations, and acknowledgments
- usernames, hostnames, cluster names, and absolute local paths
- raw private logs, receipts, cache files, and machine-generated status dumps
- unpublished datasets or files that cannot be redistributed
- credentials, tokens, API keys, and shell history

## Minimal Reproduction Plan

When we populate this repository, the goal should be to provide:

1. installation steps
2. data access or preprocessing instructions
3. one small end-to-end smoke command
4. one or two main experiment commands
5. instructions to reproduce the reported tables/figures

## Current Status

This is only the anonymous `SAVER` release skeleton.
The actual file selection, sanitization, and reproducibility pass still need
to be completed.
