# Reproducing the Experiments

## 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## 2. Models

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

`gpt2-xl` example runs:

```bash
python scripts/download_hf_model.py \
  --repo-id openai-community/gpt2-xl \
  --local-dir models/gpt2-xl
```

Local path override:

```bash
export SAVER_MODEL_NAME_OVERRIDE=/absolute/path/to/model
```

## 3. Check the Setup

```bash
python scripts/check_environment.py \
  --config configs/main/alphaedit_counterfact_qwen25_500.json
```

Main fields:

- `easyeditor_importable`
- `torch_importable`
- `transformers_importable`
- `hparams_exists`
- `dataset_exists`
- `cuda_available`

## 4. Run a Small Example

```bash
python scripts/run_example.py --config configs/examples/rome_example.json
python scripts/run_example.py --config configs/examples/memit_example.json
```

## 5. Main Sequential Editing Runs

```bash
python scripts/run_sequential_editing.py --config <config.json> --mode saver
python scripts/run_sequential_editing.py --config <config.json> --mode unconstrained
```

Mode selection:
- `--mode saver`
- `--mode unconstrained`

Write JSON output:

```bash
python scripts/run_sequential_editing.py \
  --config configs/main/alphaedit_counterfact_qwen25_500.json \
  --mode saver \
  --output outputs/alphaedit_counterfact_saver.json
```

Qwen2.5-7B 500-edit configs:

- `configs/main/alphaedit_counterfact_qwen25_500.json`
- `configs/main/alphaedit_zsre_qwen25_500.json`
- `configs/main/memit_counterfact_qwen25_500.json`
- `configs/main/memit_zsre_qwen25_500.json`
- `configs/main/rome_counterfact_qwen25_500.json`
- `configs/main/rome_zsre_qwen25_500.json`
- `configs/main/wise_counterfact_qwen25_500.json`
- `configs/main/wise_zsre_qwen25_500.json`
- `configs/main/ultraedit_counterfact_qwen25_500.json`
- `configs/main/ultraedit_zsre_qwen25_500.json`

Llama-3.1-8B 500-edit configs:

- `configs/main/alphaedit_counterfact_llama31_500.json`
- `configs/main/alphaedit_zsre_llama31_500.json`
- `configs/main/memit_counterfact_llama31_500.json`
- `configs/main/memit_zsre_llama31_500.json`
- `configs/main/wise_counterfact_llama31_500.json`
- `configs/main/wise_zsre_llama31_500.json`
- `configs/main/ultraedit_counterfact_llama31_500.json`
- `configs/main/ultraedit_zsre_llama31_500.json`

## 6. Ablations

Ablation example:

```bash
python scripts/run_sequential_editing.py \
  --config configs/ablations/alphaedit_counterfact_qwen25_500_fullsample.json \
  --mode saver
```

Ablation families:

- `*_fullsample.json`
- `*_noproxy.json`
- `*_fixedq.json`
- `*_fixedbeta.json`

## 7. Robustness Settings
Configs:

- `configs/robustness/alphaedit_counterfact_qwen25_500_overlap.json`
- `configs/robustness/alphaedit_counterfact_qwen25_500_contradictory.json`
- `configs/robustness/alphaedit_counterfact_qwen25_500_phase_shift.json`
- `configs/robustness/alphaedit_counterfact_qwen25_500_sparse_oracle.json`

Run:

```bash
python scripts/run_sequential_editing.py \
  --config configs/robustness/alphaedit_counterfact_qwen25_500_contradictory.json \
  --mode saver
```

Regenerate streams:

```bash
python scripts/generate_rq4_stress_streams.py --help
```

## 8. Checkpoint Metrics

Run:

```bash
python scripts/run_checkpoint_metrics.py \
  --config configs/main/alphaedit_counterfact_qwen25_500.json \
  --mode saver \
  --every 50 \
  --output outputs/alphaedit_counterfact_checkpoints.json
```

## 9. Representation Drift

Run:

```bash
python scripts/run_representation_drift.py \
  --config configs/main/alphaedit_zsre_qwen25_500.json \
  --mode saver \
  --output-prefix outputs/zsre_alphaedit_saver
```

## 10. Data
Included streams:

- CounterFact
- zsRE
- the hard-stream variants used in the robustness section

Rebuild:

```bash
python scripts/prepare_edit_dataset.py --help
```

## 11. Config Edits

Edit the JSON files directly to change model ids, dataset paths, or SAVER
parameters.
