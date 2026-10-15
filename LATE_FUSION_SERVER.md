# Native two-branch CRNN late fusion

This experiment preserves the earlier input-fusion implementation and all historical outputs. It adds a separate entry point for representation-level fusion.

## Design

```text
CQT 88×Tc  → Conv1d → BiGRU → attention → Linear → LayerNorm → 64-D
Rhythm 9×Tr → Conv1d → BiGRU → attention → Linear → LayerNorm → 64-D
                                                      ↓ concatenate
                                                   128-D → MLP → score
```

- CQT and rhythm keep their native time axes; no interpolation or frame-level concatenation.
- There is no fixed scalar modality weight and no hand-set 80/20 split.
- Equal-size LayerNorm projections make the fusion inputs comparable; the final MLP learns how to use them.
- The rhythm branch is smaller internally (32 conv channels, 32 hidden units per GRU direction), while both branch outputs are projected to 64-D.
- Same frozen common-v1 split and train-only normalization are used for all three ablations.

## Files

- `src/data/native_sequences.py`: immutable native sequence loader, validity masks, separate statistics.
- `src/models/model_late_fusion.py`: two independent branches and learned representation fusion.
- `src/prepare_late_fusion.py`: create train-only native normalization.
- `src/train_late_fusion.py`: train `cqt-only`, `rhythm-only`, or `late-fusion`.
- `tests/test_late_fusion.py`: unit and CLI regression tests.

## Prepare once

```bash
python -u src/prepare_late_fusion.py \
  --output datasets/features/cqt_rhythm_native_normalization.npz
```

The local cache has already been generated. `rhythm_windows/` is ignored by Git and must still be transferred separately to the server.

## Run the three ablations

Use unique output and log names. Do not enable `--evaluate-test` during development.

### CQT-only branch

```bash
set -o pipefail
python -u src/train_late_fusion.py \
  --mode cqt-only \
  --device cuda --batch-size 32 --num-workers 4 \
  --max-epochs 100 --patience 15 --min-delta 0.001 --seed 42 \
  --output outputs/cqt_rhythm_common_v1/late_fusion/cqt_only/run_01 \
  2>&1 | tee late_cqt_only_run_01.log
```

### Rhythm-only branch

```bash
set -o pipefail
python -u src/train_late_fusion.py \
  --mode rhythm-only \
  --device cuda --batch-size 32 --num-workers 4 \
  --max-epochs 100 --patience 15 --min-delta 0.001 --seed 42 \
  --output outputs/cqt_rhythm_common_v1/late_fusion/rhythm_only/run_01 \
  2>&1 | tee late_rhythm_only_run_01.log
```

### CQT + rhythm late fusion

```bash
set -o pipefail
python -u src/train_late_fusion.py \
  --mode late-fusion \
  --device cuda --batch-size 32 --num-workers 4 \
  --max-epochs 100 --patience 15 --min-delta 0.001 --seed 42 \
  --output outputs/cqt_rhythm_common_v1/late_fusion/combined/run_01 \
  2>&1 | tee late_fusion_run_01.log
```

Running all three concurrently on one GPU is not recommended: late fusion has two branches and the processes will compete for GPU memory, CPU workers, and disk I/O. Run sequentially on one GPU, or separately only when the platform has allocated independent GPUs.

## Interpretation

- `rhythm-only` establishes whether the nine rhythm channels contain predictive information by themselves.
- `cqt-only` is the same architecture family and native common-v1 data control.
- `late-fusion` tests whether independently encoded rhythm adds information beyond CQT.
- If late fusion does not beat CQT-only, the conclusion applies to this rhythm representation and fusion architecture—not to rhythm in general.
- Equal embeddings are not proof of equal causal influence; inspect ablations rather than treating learned MLP coefficients as reliable modality importance.
