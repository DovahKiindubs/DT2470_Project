# 实验输出

- **当前重跑**：`rerun_v1/`，固定共同子集，9组核心实验。尚未在本机执行正式训练。
- **旧结果**：[`../archives/pre_rerun_v1/outputs/`](../archives/pre_rerun_v1/outputs/)，完整保留所有原始run与旧台账。
- **服务器操作**：[`../RERUN_SERVER.md`](../RERUN_SERVER.md)。
- **唯一实验清单**：[`../configs/rerun_v1.json`](../configs/rerun_v1.json)。

新结果统一为：

```text
outputs/rerun_v1/
  hgb/<cqt|cqt_rhythm|cqt_structure|cqt_rhythm_structure>/seed_42_run_01/
  early/<cqt_only|cqt_rhythm>/seed_42_run_01/
  late/<cqt_only|rhythm_only|combined>/seed_42_run_01/
  summaries/RESULTS.md
  summaries/results.json
```

每个运行目录只有 `run.json`（状态、命令和版本来源）、`train.log`（日志）和 `artifacts/`（原训练器生成的模型、指标、预测与曲线记录）。不再把日志散放在根目录。

`run.json`是状态来源；汇总只是可重新生成的视图。不删除/覆盖失败运行；重试使用 `--run-name run_02`。硬关机/kill -9后可能残留running状态，需核实进程与日志，不会自动将其当完成或恢复训练。

刷新：`python scripts/run_experiments.py --summarize`。只更新当前suite的summaries，不扫描旧归档、不覆盖此README。禁止对全部历史与新结果做一个无条件总排名。
