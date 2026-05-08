# SAVER

SAVER is a pre-commit controller for sequential knowledge editing. It sits
between an editing method and a continually edited language model, evaluates a
candidate edit before it is committed, and rejects edits that would push the
model too far past the configured distortion budget.

This repository contains the code used to run SAVER with EasyEdit-supported
editors, including AlphaEdit, MEMIT, ROME, WISE, and UltraEdit. 

## What Is In This Repository

- `src/saver/`: SAVER itself
- `scripts/`: runnable entry points for experiments and analysis
- `configs/`: main runs, ablations, robustness settings, and small example runs
- `data/`: prepared edit streams and small helper assets
- `external/EasyEdit/`: vendored EasyEdit code plus the hparams used here

Large model weights are not bundled. Reviewers or users should download only
the models they plan to run.

## Setup

Create a virtual environment and install the package:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

If you plan to run Llama-based configs, make sure your Hugging Face account has
access to the model and either run `huggingface-cli login` or export
`HF_TOKEN`.

## Download a Model

The configs use public Hugging Face model ids. You can either let Transformers
resolve them directly, or download a checkpoint into a local folder first.

Example:

```bash
python scripts/download_hf_model.py \
  --repo-id Qwen/Qwen2.5-7B \
  --local-dir models/Qwen2.5-7B
```

If you want to run a local checkpoint path, set:

```bash
export SAVER_MODEL_NAME_OVERRIDE=/absolute/path/to/model
```

## Check the Environment

Before launching a long run, verify that the selected config can see its
dependencies, hparams, dataset path, and CUDA device:

```bash
python scripts/check_environment.py \
  --config configs/alphaedit_counterfact_qwen25_500.json
```

## Run the Main Experiments

The main entry point is:

```bash
python scripts/run_sequential_editing.py --config <config.json> --mode saver
python scripts/run_sequential_editing.py --config <config.json> --mode unconstrained
```

The same config file is used for both modes. `--mode saver` enables the SAVER
controller, while `--mode unconstrained` runs the underlying editor without the
pre-commit gate.

For example:

```bash
python scripts/run_sequential_editing.py \
  --config configs/alphaedit_counterfact_qwen25_500.json \
  --mode saver
```

To save the summary instead of only printing it:

```bash
python scripts/run_sequential_editing.py \
  --config configs/alphaedit_counterfact_qwen25_500.json \
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

## More Detailed Run Instructions

The full reproduction guide, including model download commands, main config
lists, ablations, checkpoint metrics, and representation-drift runs, is in
[docs/reproduce.md](docs/reproduce.md).
