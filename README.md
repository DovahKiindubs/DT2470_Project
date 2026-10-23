# Piano Performance Difficulty Prediction

This is a DT2470 Music Informatics course project. It uses the PSyllabus solo-piano dataset to predict ordinal performance difficulty on a 0–10 scale and investigates whether explicit rhythm- and structure-related features improve upon a CQT baseline.

## Methods

- **HGB regression:** compares fixed-length CQT, rhythm, and structure features and their combinations.
- **Early CRNN:** aligns an 88-channel CQT sequence with a 9-channel rhythm sequence and concatenates them at the input level.
- **Late Fusion CRNN:** encodes CQT and rhythm in separate branches before fusing their learned representations.
- **Primary metric:** Kendall's Tau-c; Spearman correlation, MAE, MSE, and Acc±1 are also reported.

All experiments use the same frozen split: 4,089 training pieces, 2,040 validation pieces, and 1,557 test pieces, with random seed 42. On the final test set, HGB with CQT, rhythm, and structure features achieved the highest Tau-c of 0.7204.

## Environment

Python 3.12 is recommended. Create a virtual environment in the project root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For CUDA systems, install a PyTorch build compatible with the server's CUDA version instead of copying `.venv` from another machine. `requirements.txt` is a snapshot of the local development environment and may require version adjustments on Python 3.11 or a different CUDA setup.

## Data Preparation

See [`datasets/README.md`](datasets/README.md) for the data layout and file descriptions. A complete reproduction requires:

- `datasets/features/cqt/*.bin`
- `datasets/features/rhythm_windows/*.npz` and `index.npz`
- `datasets/features/cqt_hard_windows.npz`
- `datasets/features/rhythm_stats.npz`
- `datasets/features/structure_stats.npz`
- both normalization caches, metadata, the frozen split, and the manifest

The final ten experiments do not require the original audio. Audio is needed only when extracting rhythm or structure features again.

## Reproducing the Experiments

Run all commands from the project root and explicitly select the final experiment configuration:

```bash
# Print the commands for all ten experiments without training
python scripts/run_experiments.py --config configs/rerun_v2.json --dry-run

# Validate files, split membership, and hashes without training
python scripts/run_experiments.py --config configs/rerun_v2.json --check

# Run all ten experiments sequentially
python scripts/run_experiments.py --config configs/rerun_v2.json \
  --execute --device cuda --num-workers 4 --threads 4 \
  --seed 42 --run-name run_01

# Summarize validation results
python scripts/run_experiments.py --config configs/rerun_v2.json --summarize
```

Use `--group hgb`, `--group early`, or `--group late` to run only one group. Results are written to `outputs/rerun_v2/`; an existing run directory is never overwritten.

After all models have been frozen, run the final test evaluation once:

```bash
python scripts/evaluate_test.py --config configs/rerun_v2.json \
  --execute --acknowledge-final-test FROZEN_MODELS_FINAL_TEST \
  --device cuda --batch-size 32 --num-workers 4 \
  --seed 42 --run-name run_01
```

The test script loads the saved models for inference only. It does not retrain models or select hyperparameters.

## Project Structure

```text
configs/   Final experiment catalogue
src/       Feature, data, model, training, and evaluation code
scripts/   Experiment, test, and case-analysis entry points
tests/     Unit tests and synthetic-data regression tests
datasets/  Metadata, splits, feature caches, and local large files
outputs/   Experiment results generated at runtime
```

See [`src/README.md`](src/README.md) for a source-code overview.
