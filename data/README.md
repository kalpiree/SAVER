# Data

This directory contains:

- prepared CounterFact and zsRE stream files used by the released configs
- prepared hard-stream variants for the robustness experiments
- small helper assets for environment checks and quick sanity runs

Large raw datasets and model checkpoints are not bundled here.

To rebuild stream files from raw public datasets, use:

```bash
python scripts/prepare_edit_dataset.py --help
```
