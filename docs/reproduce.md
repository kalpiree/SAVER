# Reproducing the Experiments

This note is meant to be practical. It explains how to set up the repository,
download only the assets you need, and run the main experiment families from
the paper.

## 1. Environment

Create a virtual environment and install the package:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

The repository already includes the EasyEdit source used by the editor wrapper,
so there is no separate EasyEdit installation step.

For the main runs, use a machine with a CUDA-capable GPU and enough memory for
the selected model.

## 2. Models

This repository does not bundle model checkpoints. Download only the models you
plan to use.

Qwen2.5-7B:

```bash
python scripts/download_hf_model.py \
  --repo-id Qwen/Qwen2.5-7B \
  --local-dir models/Qwen2.5-7B
```

Llama-3.1-8B-Instruct:

```bash
python scripts/download_hf_model.py \
  --repo-id meta-llama/Llama-3.1-8B-Instruct \
  --local-dir models/Llama-3.1-8B-Instruct
```

Small example runs with `gpt2-xl`:

```bash
python scripts/download_hf_model.py \
  --repo-id openai-community/gpt2-xl \
  --local-dir models/gpt2-xl
```

If you use a gated model such as Llama, log in with Hugging Face first, or set
`HF_TOKEN` in the environment.

If you prefer to use a local path instead of the model id stored in a config,
set:

```bash
export SAVER_MODEL_NAME_OVERRIDE=/absolute/path/to/model
```

## 3. Check the Setup

Before launching a long run, check that the config can see its model, dataset,
hparams file, and CUDA device:

```bash
python scripts/check_environment.py \
  --config configs/alphaedit_counterfact_qwen25_500.json
```

The script prints a JSON status report. The main fields to look at are:

- `easyeditor_importable`
- `torch_importable`
- `transformers_importable`
- `hparams_exists`
- `dataset_exists`
- `cuda_available`

## 4. Run a Small Example

If you want to confirm the full SAVER execution path before starting a larger
job, run one of the included examples:

```bash
python scripts/run_example.py --config configs/rome_example.json
python scripts/run_example.py --config configs/memit_example.json
```

These are not separate toy scripts; they run the same SAVER path used by the
main experiments, just on a much smaller stream.

## 5. Main Sequential Editing Runs

The main experiment driver is:

```bash
python scripts/run_sequential_editing.py --config <config.json> --mode saver
python scripts/run_sequential_editing.py --config <config.json> --mode unconstrained
```

There is no separate "SAVER config" and "unconstrained config". The same config
file is used in both cases; `--mode` selects whether the pre-commit controller
is active.

By default the script prints a JSON summary to stdout. To save it:

```bash
python scripts/run_sequential_editing.py \
  --config configs/alphaedit_counterfact_qwen25_500.json \
  --mode saver \
  --output outputs/alphaedit_counterfact_saver.json
```

Included Qwen2.5-7B 500-edit configs:

- `configs/alphaedit_counterfact_qwen25_500.json`
- `configs/alphaedit_zsre_qwen25_500.json`
- `configs/memit_counterfact_qwen25_500.json`
- `configs/memit_zsre_qwen25_500.json`
- `configs/rome_counterfact_qwen25_500.json`
- `configs/rome_zsre_qwen25_500.json`
- `configs/wise_counterfact_qwen25_500.json`
- `configs/wise_zsre_qwen25_500.json`
- `configs/ultraedit_counterfact_qwen25_500.json`
- `configs/ultraedit_zsre_qwen25_500.json`

Included Llama-3.1-8B 500-edit configs:

- `configs/alphaedit_counterfact_llama31_500.json`
- `configs/alphaedit_zsre_llama31_500.json`
- `configs/memit_counterfact_llama31_500.json`
- `configs/memit_zsre_llama31_500.json`
- `configs/wise_counterfact_llama31_500.json`
- `configs/wise_zsre_llama31_500.json`
- `configs/ultraedit_counterfact_llama31_500.json`
- `configs/ultraedit_zsre_llama31_500.json`

## 6. Ablations

The ablation runs use the same driver. For example:

```bash
python scripts/run_sequential_editing.py \
  --config configs/alphaedit_counterfact_qwen25_500_fullsample.json \
  --mode saver
```

Included ablation families:

- `*_fullsample.json`
- `*_noproxy.json`
- `*_fixedq.json`
- `*_fixedbeta.json`

The included ablations are provided for the AlphaEdit-based canonical setting.

## 7. Robustness Settings

The repository includes the hard-stream settings used for the robustness
analysis:

- `configs/alphaedit_counterfact_qwen25_500_overlap.json`
- `configs/alphaedit_counterfact_qwen25_500_contradictory.json`
- `configs/alphaedit_counterfact_qwen25_500_phase_shift.json`
- `configs/alphaedit_counterfact_qwen25_500_sparse_oracle.json`

Run them with the same main driver:

```bash
python scripts/run_sequential_editing.py \
  --config configs/alphaedit_counterfact_qwen25_500_contradictory.json \
  --mode saver
```

If you want to rebuild the hard streams instead of using the prepared files:

```bash
python scripts/generate_rq4_stress_streams.py --help
```

## 8. Checkpoint Metrics

To reproduce intermediate ESR/PSR/NSR measurements during a run:

```bash
python scripts/run_checkpoint_metrics.py \
  --config configs/alphaedit_counterfact_qwen25_500.json \
  --mode saver \
  --every 50 \
  --output outputs/alphaedit_counterfact_checkpoints.json
```

## 9. Representation Drift

The representation-drift script saves hidden-state snapshots for later plotting:

```bash
python scripts/run_representation_drift.py \
  --config configs/alphaedit_zsre_qwen25_500.json \
  --mode saver \
  --output-prefix outputs/zsre_alphaedit_saver
```

This writes a JSON metadata file and an `.npz` tensor bundle with the requested
checkpoints.

## 10. Data

Prepared stream files are already included for:

- CounterFact
- zsRE
- the hard-stream variants used in the robustness section

If you want to rebuild a SAVER-compatible stream from a public raw dataset:

```bash
python scripts/prepare_edit_dataset.py --help
```

## 11. Notes

- The repository intentionally excludes cached model weights, generated result
  files, cluster wrappers, and monitoring-baseline code.
- Some model families have different memory requirements; if a config fails to
  load on your machine, start with the example run or switch to a smaller model
  first.
- The configs can be edited directly if you want to change the model id,
  dataset path, or SAVER parameters instead of using command-line overrides.
