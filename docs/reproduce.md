# Reproducing the Experiments

Run the commands from the repository root. Experiment configurations are in
`configs/main/`, `configs/ablations/`, and `configs/robustness/`; all launchers
are in `scripts/`. Prepare the data in Section 10 before running experiments.

## 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Use a CUDA-enabled environment for editor-backed model runs. The sentence
encoder defaults to `sentence-transformers/all-MiniLM-L6-v2` on CPU and is
downloaded on first use. Set `embedding_model` to a local directory in the
experiment configuration for offline use.

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

Qwen3-8B-Base:

```bash
python scripts/download_hf_model.py \
  --repo-id Qwen/Qwen3-8B-Base \
  --local-dir models/Qwen3-8B-Base
```

Qwen3.5-9B:

```bash
python scripts/download_hf_model.py \
  --repo-id Qwen/Qwen3.5-9B \
  --local-dir models/Qwen3.5-9B
```

`gpt2-xl` example runs:

```bash
python scripts/download_hf_model.py \
  --repo-id openai-community/gpt2-xl \
  --local-dir models/gpt2-xl
```

Optional local path override; unset it before switching to another backbone:

```bash
export SAVER_MODEL_NAME_OVERRIDE=/absolute/path/to/model
```

Qwen3.5 AlphaEdit and UltraEdit settings are in `hparams/`. If the installed
Transformers version cannot load the Qwen3.5 configuration, the optional
overlay can be installed and activated in a separate Qwen3.5 environment:

```bash
bash scripts/install_transformers_overlay.sh
export PYTHONPATH="${PWD}/compat/qwen35:${PWD}/external/transformers_src${PYTHONPATH:+:${PYTHONPATH}}"
```

The helper installs unpinned upstream Transformers and a prerelease tokenizer
dependency. Record the installed versions and revision with the run.

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

Run both methods with the same configuration and save their summaries:

```bash
python scripts/run_sequential_editing.py \
  --config configs/main/alphaedit_counterfact_qwen25_500.json \
  --mode saver \
  --output outputs/alphaedit_counterfact_saver.json

python scripts/run_sequential_editing.py \
  --config configs/main/alphaedit_counterfact_qwen25_500.json \
  --mode unconstrained \
  --output outputs/alphaedit_counterfact_unconstrained.json
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

NoProxy matches the mean sampling probability of a completed SAVER run on
the same dataset path, seed, and requested stream length:

```bash
python scripts/run_sequential_editing.py \
  --config configs/ablations/alphaedit_counterfact_qwen25_500_noproxy.json \
  --mode saver \
  --reference-run outputs/alphaedit_counterfact_saver.json \
  --output outputs/alphaedit_counterfact_noproxy.json
```

Use the reference output from Section 5. This matches the expected sampling
budget, not the realized number of sampled rounds. Fixed-boundary
configurations continue to the next request after a rejection.

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

Regenerate streams from the rebuilt CounterFact bank before running:

```bash
python scripts/generate_rq4_stress_streams.py \
  --source data/streams/counterfact_full.jsonl \
  --output-dir data/streams \
  --limit 500 \
  --seed 17
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

The experiment runners require `probe_schema_version: 2` in every record.
Rebuild older streams from their raw dataset sources with the current
converters; adding the version field alone does not correct probe targets.
Keep the resulting stream fixed across compared methods.

Prepare the full CounterFact bank for the main runs and probe ablations:

```bash
python scripts/prepare_hf_counterfact_stream.py \
  --output data/streams/counterfact_full.jsonl \
  --limit 0 \
  --seed 17 \
  --max-locality 16
```

The converter defaults to `azhx/counterfact`, combines its `train,test`
splits, shuffles with the supplied seed, and keeps all records when
`--limit 0` is used. Override `--dataset` and `--splits` for another source.

Prepare the stream files referenced by the long-stream configurations:

```bash
python scripts/prepare_hf_counterfact_stream.py \
  --output data/streams/counterfact_rebuttal_2000.jsonl \
  --limit 2000 \
  --seed 17

python scripts/prepare_hf_counterfact_stream.py \
  --output data/streams/counterfact_rebuttal_5000.jsonl \
  --limit 5000 \
  --seed 17
```

For a local raw zsRE file:

```bash
ZSRE_RAW=/absolute/path/to/zsre.json
python scripts/prepare_edit_dataset.py \
  --input "${ZSRE_RAW}" \
  --output data/streams/zsre_full.jsonl \
  --format zsre \
  --shuffle \
  --seed 17 \
  --max-locality 16
```

The same converter accepts local CounterFact data with `--format counterfact`.
Portability questions use their own targets and remain separate from
same-target paraphrases. Locality reference tokens are obtained from the
initial model before editing.

## 11. Config Edits

Edit the JSON files directly to change model ids, dataset paths, or SAVER
parameters. Keep the model, editor settings, stream order, seed, and
evaluation corpus fixed across comparisons. The runners use the
`ppl_text_path` in the configuration unless `--ppl-text-path` is supplied.

## 12. Long Streams and Qwen3.5

Qwen3.5-9B with AlphaEdit defaults to 2,000 submitted edits:

```bash
bash scripts/run_qwen35_alphaedit.sh
```

Qwen3.5-9B with UltraEdit defaults to 1,000 submitted edits:

```bash
bash scripts/run_qwen35_ultraedit.sh
```

Both default to SAVER. Set `MODE=unconstrained` to run without SAVER. For
5,000 edits with evaluations at 500, 1,000, 2,500, and 5,000, each comparison
launcher runs the unconstrained method followed by SAVER:

```bash
bash scripts/run_qwen35_alphaedit_5000_comparison.sh
bash scripts/run_qwen35_ultraedit_5000_comparison.sh
```

Choose a shorter run and evaluation schedule explicitly:

```bash
LIMIT=1000 MIN_EDITS=1000 CHECKPOINTS=500,1000 \
  bash scripts/run_qwen35_alphaedit.sh
```

`LIMIT` and `CHECKPOINTS` count submitted requests, including rejections.
`MIN_EDITS` checks the available stream size; it is not an acceptance target.
The runner also evaluates at the final submitted step. To request 5,000
edits, use a configuration pointing to a stream with at least 5,000 records.

Qwen3-8B-Base with UltraEdit:

```bash
CONFIG=configs/main/ultraedit_counterfact_qwen3_1000.json \
LIMIT=1000 MIN_EDITS=1000 CHECKPOINTS=500,1000 \
  bash scripts/run_stream.sh
```

For two visible GPUs, select
`configs/main/ultraedit_counterfact_qwen35_1000_dualgpu.json` with the same
launcher. GPU availability and process scheduling are managed outside these
scripts.

## 13. Acceptance-Matched Baselines

Run SAVER, random rejection, the current-probe gate, and the KL-drift gate
on the Qwen2.5-7B AlphaEdit configuration:

```bash
OUTPUT_DIR=outputs/matched_baselines \
  bash scripts/run_matched_baseline_queue.sh
```

The queue runs SAVER first and passes its final acceptance rate to the three
baselines. Random rejection uses that rate as a Bernoulli acceptance
probability. The probe and KL gates combine running score ranks with a
cumulative acceptance budget. Their realized acceptance rates can differ
from SAVER's. Report `run_summary.acceptance_rate` for each run.

Override `CONFIG`, `LIMIT`, `MIN_EDITS`, and `CHECKPOINTS` together to change
the comparison. The available runner modes are `saver`, `unconstrained`,
`random_reject`, `probe_gate`, and `kl_gate`.

## 14. Probe Quantity

Run SAVER with 25%, 50%, 75%, and 100% of the online locality probes:

```bash
OUTPUT_DIR=outputs/probe_coverage \
  bash scripts/run_probe_coverage_queue.sh
```

The default is Qwen2.5-7B with AlphaEdit for 500 submitted edits, evaluated
at 100, 250, and 500. The locality bank is split into online and held-out
halves before editing. Only the online locality subset varies; generality
probes and the held-out bank remain fixed.

The queue filters for at least eight locality prompts per edit before
applying the stream limit. It requires 500 eligible records by default and
rejects online/audit prompt overlap and indistinguishable coverage levels.

## 15. Probe Quality

Relevant and weak locality monitoring with Qwen2.5-7B and AlphaEdit:

```bash
EDITOR=alphaedit OUTPUT_DIR_BASE=outputs/probe_quality_alphaedit \
  bash scripts/run_probe_quality_queue.sh
```

Use `EDITOR=ultraedit` for the UltraEdit configuration. Both conditions use
500 submitted edits, checkpoints at 100, 250, and 500, the same seeded
locality split, and unchanged generality probes. Weak probes use unrelated
donors while evaluation uses the fixed relevant held-out bank.

Donor selection prefers low-similarity, length-matched probes and checks
base-model correctness. When those filters leave too few donors, the
implementation relaxes them and records `weak_fallback_count` in
`probe_quality_info`; inspect that count when reporting the experiment.
Audit prompts remain excluded. Insufficient disjoint donors cause an error.

## 16. Controlled-Risk Detection Delay

Create a full-evaluation Qwen2.5-7B source run:

```bash
python scripts/run_sequential_editing.py \
  --config configs/ablations/alphaedit_counterfact_qwen25_500_fullsample.json \
  --mode saver \
  --output outputs/alphaedit_counterfact_fullsample.json
```

Replay its joint boundary-risk vectors under controlled risk shifts:

```bash
python scripts/run_detection_delay.py \
  --source outputs/alphaedit_counterfact_fullsample.json \
  --output outputs/detection_delay \
  --beta 0.99 \
  --theta 0.10 \
  --alpha 0.10 \
  --q-min 1 \
  --pre-steps 100 \
  --post-steps 2000 \
  --replications 2000 \
  --shifts 0.10,0.20,0.30 \
  --seed 17
```

The source must contain `snapshots` with risks for every boundary in the
chosen grid and controlled mean risk at the fixed boundary before the
change. Use the output of `run_sequential_editing.py`, not the stream
runner's `records` output. The default grid is 0.55 to 0.99 in steps of 0.02;
set `--beta-grid` and `--grid-size` together when using another grid.

The diagnostic replays fully evaluated risk vectors without editing model
weights. The JSON and CSV outputs report detection rates, delay summaries,
censoring, and simulated pre-alarm excess-risk exposure. Delay summaries
are conditional on detection; report the detection rate and censored count
alongside them.

## 17. Outputs and Execution

`run_stream.sh` writes checkpoint summaries under `OUTPUT_DIR/json/` and
per-request events under `OUTPUT_DIR/events/`. The JSON `records` contain
acceptance, ESR, PSR, all-request ESR/PSR, NSR, PPL, and risk summaries;
probe-split runs also include held-out metrics and monitor/audit correlation.
The current token-based metrics use the first target token. All-request
ESR and PSR assign zero success to rejected requests.

Stream checkpoint JSON is updated during the run. These are metric
checkpoints, not resumable model-weight checkpoints. The launchers run in
the foreground, execute queued conditions sequentially, and stop on an
error. They do not provide automatic GPU allocation, OOM retries, or
resume support. Use the execution environment's scheduler or session
manager for unattended runs.

`OUTPUT_DIR`, `RUN_STAMP`, `QUEUE_STAMP`, and `PYTHON_BIN` control output
locations, run names, and the Python interpreter. Use a distinct output
location or run name for each independent invocation.
