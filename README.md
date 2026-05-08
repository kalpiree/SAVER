# SAVER

SAVER is a pre-commit controller for sequential knowledge editing.

## What Is In This Repository

- `src/saver/`: SAVER itself
- `scripts/`: runnable entry points for experiments and analysis
- `configs/main/`: main runs
- `configs/ablations/`: ablations
- `configs/robustness/`: robustness settings
- `configs/examples/`: small example runs
- `data/`: prepared edit streams and small helper assets
- `external/EasyEdit/`: vendored EasyEdit code plus the hparams used here

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Download a Model

```bash
python scripts/download_hf_model.py \
  --repo-id Qwen/Qwen2.5-7B \
  --local-dir models/Qwen2.5-7B
```

Local path override:

```bash
export SAVER_MODEL_NAME_OVERRIDE=/absolute/path/to/model
```

## Check the Environment

```bash
python scripts/check_environment.py \
  --config configs/main/alphaedit_counterfact_qwen25_500.json
```

## Run a Small End-to-End Example

```bash
python scripts/run_example.py --config configs/examples/rome_example.json
python scripts/run_example.py --config configs/examples/memit_example.json
```

## Run the Main Experiments

```bash
python scripts/run_sequential_editing.py --config <config.json> --mode saver
python scripts/run_sequential_editing.py --config <config.json> --mode unconstrained
```

Mode selection:
- `--mode saver`
- `--mode unconstrained`

```bash
python scripts/run_sequential_editing.py \
  --config configs/main/alphaedit_counterfact_qwen25_500.json \
  --mode saver
```

Write JSON output:

```bash
python scripts/run_sequential_editing.py \
  --config configs/main/alphaedit_counterfact_qwen25_500.json \
  --mode saver \
  --output outputs/alphaedit_counterfact_saver.json
```

## Included Experiment Families

Main 500-edit configs are included for:

- AlphaEdit
- MEMIT
- ROME
- WISE
- UltraEdit

AlphaEdit ablation configs are included for:

- full-sample monitoring
- no-proxy sampling
- fixed-q sampling
- fixed-beta boundary selection

Robustness configs are included for:

- contradictory streams
- overlap-heavy streams
- easy-to-hard phase shifts
- sparse oracle evaluation

## Reproduction Guide

[docs/reproduce.md](docs/reproduce.md)
