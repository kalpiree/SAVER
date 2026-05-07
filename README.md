# SAVER

SAVER is a pre-commit controller for sequential knowledge editing. It monitors
candidate edits before they are committed to a continually edited language
model, with the goal of preserving edit utility while reducing cumulative
distortion over long edit streams.

## Repository Contents

- `src/`: core implementation
- `scripts/`: experiment drivers, evaluation utilities, and analysis scripts
- `configs/`: experiment configuration files
- `data/`: dataset metadata, stream definitions, and preprocessing helpers
- `docs/`: supplementary usage and experiment notes
- `results/`: tables, summaries, and derived outputs
- `tests/`: smoke tests and regression checks
- `external/`: third-party dependencies or wrappers included with the release
- `assets/`: static figures or media used in documentation

## Scope

The repository is organized to support:

- long-run sequential editing experiments
- risk-control comparisons
- monitoring-efficiency comparisons
- hard-stream robustness settings
- component ablations

## Workflow

The intended workflow is:

1. install dependencies
2. prepare model checkpoints and datasets
3. select an experiment configuration from `configs/`
4. launch runs from `scripts/`
5. collect outputs and summaries from `results/`

## Reproduction

The release is structured so that the main reported experiments can be
reproduced from the included scripts and configuration files. Typical usage is
through experiment-specific drivers in `scripts/` together with the matching
configuration files in `configs/`.

## Directory Summary

- `configs/` defines experiment settings
- `scripts/` contains runnable entry points
- `src/` contains reusable implementation code
- `results/` stores processed outputs used for reporting

