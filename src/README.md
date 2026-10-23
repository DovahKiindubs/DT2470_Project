# Source Code Guide

The source code is organized as feature extraction → data handling → models → training → experiment orchestration. Some top-level files are compatibility entry points; the actual implementations live in the subdirectories and should not be removed without checking their callers.

## Directory Responsibilities

- `features/`
  - `cqt.py`: CQT loading and global statistics.
  - `cqt_windows.py`: sliding-window and high-activity-window pooling.
  - `rhythm.py`: 410-dimensional rhythm summary features.
  - `rhythm_windows.py`: 9-channel temporal rhythm features.
  - `structure.py`: 73-dimensional structure-related statistics derived from chroma self-similarity matrices.
- `data/`
  - Metadata, split, and hash validation.
  - Feature concatenation for HGB.
  - Temporal alignment for Early Fusion and paired inputs for Late Fusion.
  - Training-set normalization and audio preprocessing.
- `models/`
  - `model_crnn.py`: Conv1D, BiGRU, masked attention, and regression head.
  - `model_late_fusion.py`: separate CQT and rhythm branches with representation-level fusion.
- `training/`
  - `hgb.py`: Histogram-based Gradient Boosting regression.
  - `crnn.py`: Early Fusion and single-input CRNN training.
  - `late_fusion.py`: two-branch Late Fusion training.
  - `ridge.py`: early linear baseline retained for traceability.
- `reporting/metrics.py`: Tau-c, Spearman, MAE, MSE, and Acc±1.
- `experiments/suite.py`: orchestration, input checks, run-state tracking, and summaries for all ten experiments.
- `experiments/evaluate_test.py`: final test-only inference for frozen models.

## Recommended Entry Points

Run these commands from the project root:

```bash
python scripts/run_experiments.py --config configs/rerun_v2.json --dry-run
python scripts/run_experiments.py --config configs/rerun_v2.json --check
python scripts/run_experiments.py --config configs/rerun_v2.json --execute --group hgb
python scripts/run_experiments.py --config configs/rerun_v2.json --execute --group early
python scripts/run_experiments.py --config configs/rerun_v2.json --execute --group late
python scripts/run_experiments.py --config configs/rerun_v2.json --summarize
```

To run the layered modules directly:

```bash
PYTHONPATH=src python -m training.crnn --help
PYTHONPATH=src python -m training.late_fusion --help
PYTHONPATH=src python -m training.hgb --help
```

## Compatibility Entry Points

The following top-level files forward to the layered implementations and are retained for compatibility with older tests and commands:

```text
train.py / train_hgb.py / train_crnn.py / train_late_fusion.py
features_cqt.py / features_cqt_windows.py / features_rhythm.py
features_rhythm_temporal.py / features_structure.py
prepare_crnn.py / prepare_late_fusion.py / preprocess_audio.py
summarize_experiments.py
```

## Experimental Scope

- The final configuration is `configs/rerun_v2.json`, containing four HGB, three Early, and three Late experiments.
- All experiments share the frozen common subset and use random seed 42.
- HGB uses fixed-length CQT, rhythm, and structure statistics.
- The CRNN models use CQT and rhythm sequences that preserve temporal order. The current structure representation is a 73-dimensional piece-level summary, so it is not fed directly into the CRNN.
- The default training workflow evaluates the validation set only. The test set is evaluated with a separate script after model freezing.
- Performance differences between Early and Late Fusion also reflect differences in time grids, branch capacity, and prediction heads; they cannot be attributed entirely to the fusion point.
