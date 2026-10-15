# Experiment log

This log records what each experiment tested, not just which score was highest. Metrics in different split families are not directly comparable. All listed completed results use validation only; the test set remains sealed.

## Timeline and questions

### E01 — Global CQT summaries

- Inputs: 88 CQT-bin means + 88 standard deviations (176-D).
- Models: Ridge, HistGradientBoosting.
- Question: Can whole-recording spectral summaries predict ordinal piano difficulty?
- Result family: full strict split, validation N=2095.
- Outcome: HGB exceeded Ridge, but global pooling discarded local difficult passages.
- Paths: `outputs/cqt_only/`.

### E02 — Hard-window CQT pooling

- Inputs: global CQT plus mean/max statistics from high-flux 10-second windows (528-D).
- Models: Ridge, HGB.
- Question: Does emphasizing locally active/difficult-looking material help?
- Result family: full strict split, validation N=2095.
- Outcome: clear validation improvement over global CQT summaries.
- Paths: `outputs/cqt_hard_windows/`.

### E03 — Full-sequence CQT CRNN

- Input: native 88×T CQT sequence.
- Model: Conv1d ×2 → BiGRU → masked attention → regression.
- Question: Can learned temporal aggregation outperform manually pooled statistics?
- Result family: full strict split, validation N=2095.
- Best recorded result: Tau-c 0.6806, MSE 3.1493, Acc±1 0.6353.
- Path: `outputs/cqt_crnn/a800_run_01/`.

### E04 — Audio-subset engineered-feature ablations

- Inputs tried: global rhythm, richer 410-D rhythm, pooled temporal rhythm, chroma-SSM structure, hard-window CQT, and combinations.
- Models: Ridge, HGB.
- Question: Which explicitly designed musical descriptors add information beyond CQT summaries?
- Result family: audio subset, validation N=2044.
- Outcome: structure features produced a larger lift than rhythm alone in these tabular experiments; best recorded HGB combination was hard-window CQT + rhythm + structure (Tau-c 0.7034). This is validation evidence, not a test result.
- Paths: `outputs/audio_subset/`; detailed source summary: `outputs/summaries/audio_subset/feature_comparison_validation.json`.

### E05 — Temporal rhythm sequence, early/input fusion

- Inputs: CQT cropped to rhythm-center coverage; 9 rhythm channels linearly interpolated onto the CQT grid; concatenated into 97×T.
- Model: the same single CRNN as CQT-only.
- Question: Does explicit local rhythm improve the sequence model when fused at input level?
- Result family: frozen common-v1 split, validation N=2040.
- 30-epoch recorded result: Tau-c 0.6780, MSE 2.8293, Acc±1 0.6534.
- 100-epoch screenshot result: aligned CQT-only Tau-c 0.6957 versus early fusion 0.6891; no demonstrated gain from this fusion design.
- Interpretation: this rejects the tested *input-level interpolation and concatenation design*, not rhythm information in general.
- Path: `outputs/cqt_rhythm_common_v1/cqt_rhythm/run_01/`. Copy future 100-epoch directories under their documented run paths and refresh the registry.

### E06 — Planned native two-branch representation fusion

- Inputs: native 88×Tc CQT and native 9×Tr rhythm, with separate train-only normalization and no temporal interpolation.
- Model: independent Conv1d-BiGRU-attention branches; each pooled representation is projected to the same embedding size and LayerNorm-ed; embeddings are concatenated and a learned MLP predicts difficulty.
- Fixed modality weights: **none**. The fusion head learns from the two equal-size normalized representations.
- Required ablations on the same frozen common-v1 split:
  1. `cqt-only`
  2. `rhythm-only`
  3. `late-fusion`
- Status: code implemented and local smoke tests required before server training.
- Intended paths:
  - `outputs/cqt_rhythm_common_v1/late_fusion/cqt_only/run_01`
  - `outputs/cqt_rhythm_common_v1/late_fusion/rhythm_only/run_01`
  - `outputs/cqt_rhythm_common_v1/late_fusion/combined/run_01`

## Reporting checklist

For every future run, retain:

- `run_parameters.json`
- `config.json`
- `history.json` and `history.jsonl`
- `metrics.json`
- `predictions_val.json`
- `best_checkpoint.pt` and `model.pt`
- unique output directory and log file

Record in the report:

1. research question and feature representation;
2. split hash and validation N;
3. model and normalization;
4. selection/early-stopping rule;
5. Tau-c as primary metric plus Spearman, MAE, MSE, Acc±1;
6. negative results and plausible limitations;
7. test metrics only after the final design is frozen.
