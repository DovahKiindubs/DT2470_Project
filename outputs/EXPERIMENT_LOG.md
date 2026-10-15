# Experiment log

本台账保留历史研究顺序。当前报告取舍见 [REPORT_SELECTION.md](REPORT_SELECTION.md)：A组传统特征4条＋B组late三条；early100轮负结果保留但待同步完整文件。本次只更新记录和导航，旧代码和实验不丢弃。

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
- Interpretation: no gain was observed in this single reported comparison; this does not reject input fusion in general or rhythm information. The two 100-epoch run folders have not been located locally; treat their metrics as screenshot-reported, not file-reverified.
- Path: `outputs/cqt_rhythm_common_v1/cqt_rhythm/run_01/` is the 30-epoch run only. Preserve it; do not relabel it as the 100-epoch result.

### E06 — Native two-branch representation fusion（已同步结果）

- Inputs: native 88×Tc CQT and native 9×Tr rhythm, with separate train-only normalization and no temporal interpolation.
- Model: independent Conv1d-BiGRU-attention branches; each pooled representation is projected to the same embedding size and LayerNorm-ed; embeddings are concatenated and a learned MLP predicts difficulty.
- Fixed modality weights: **none**. The fusion head learns from the two equal-size normalized representations.
- Completed ablations: same frozen common-v1 split (validation N=2040), seed=42, max_epochs=100, patience=15, min_delta=0.001.
  1. `cqt-only`: Tau-c 0.6451, MSE 3.3092, Acc±1 0.6328; best61, stopped76.
  2. `rhythm-only`: Tau-c 0.6331, MSE 3.4571, Acc±1 0.5907; best61, stopped76.
  3. `late-fusion`: Tau-c 0.6841, MSE 2.7423, Acc±1 0.6863; best71, stopped86.
- Status: server results copied locally; validation IDs and five metrics checked from predictions. No test evaluation.
- Outcome: combined exceeds its corresponding mono-branch controls in this run. It does not exceed the earlier screenshot CQT-only Tau-c 0.6957; the new mono model has a changed projection/head, input coverage and normalization. Capacity and seed variability remain confounds. Do not report statistically significant or universally superior fusion.
- Paths:
  - `outputs/cqt_rhythm_common_v1/late_fusion/cqt_only/run_01`
  - `outputs/cqt_rhythm_common_v1/late_fusion/rhythm_only/run_01`
  - `outputs/cqt_rhythm_common_v1/late_fusion/combined/run_01`

## 本次核验补记

- 核心7条的预测ID唯一且匹配相应当前validation成员，5项指标复算一致。
- E04中A1旧HGB基线的split SHA尚未从Git历史复现；虽然验证成员一致、配置样本数相同，不能据此断言训练成员完全相同。A2–A4的SHA与当前split的CRLF版本对应。补证前将A组标作探索性，而不是严格受控比较。
- E04旧句“structure lift更大”只适用于指定410维rhythm对照；36维池化节奏版本更高，不应泛化为结构总优于节奏。结构统计未经音乐实例验证，当前novelty函数不是完整Foote checkerboard，不声称检测到了真实乐曲段落。
- 全量38个metrics运行保留；当前README已纳入late三组，summaries中的35条机器索引仍为历史快照。

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
