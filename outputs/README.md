# 实验输出

- **当前重跑配置**：`rerun_v2/`，固定共同子集，10组（HGB4、Early3、Late3）；本轮新增Early Rhythm-only，正式结果待服务器运行生成。
- **上一轮9组结果**：`rerun_v1/`，原样保留，不与v2混写。
- **更早结果**：[`../archives/pre_rerun_v1/outputs/`](../archives/pre_rerun_v1/outputs/)，完整保留所有原始run与旧台账。
- **服务器操作**：[`../RERUN_SERVER.md`](../RERUN_SERVER.md)，优先看顶部v2部分。
- **本轮实验清单**：[`../configs/rerun_v2.json`](../configs/rerun_v2.json)。旧清单 `configs/rerun_v1.json` 不变。

新结果统一为：

```text
outputs/rerun_v2/
  hgb/<cqt|cqt_rhythm|cqt_structure|cqt_rhythm_structure>/seed_42_run_01/
  early/<cqt_only|rhythm_only|cqt_rhythm>/seed_42_run_01/
  late/<cqt_only|rhythm_only|combined>/seed_42_run_01/
  summaries/RESULTS.md
  summaries/results.json
```

每个运行目录只有 `run.json`（状态、命令和版本来源）、`train.log`（日志）和 `artifacts/`（原训练器生成的模型、指标、预测与曲线记录）。不再把日志散放在根目录。

`run.json`是状态来源；汇总只是可重新生成的视图。不删除/覆盖失败运行；重试使用 `--run-name run_02`。硬关机/kill -9后可能残留running状态，需核实进程与日志，不会自动将其当完成或恢复训练。

刷新v2：`python scripts/run_experiments.py --config configs/rerun_v2.json --summarize`。只更新v2的summaries，不扫描v1或归档、不覆盖此README。省略`--config`仍指旧v1。禁止对全部历史与新结果做一个无条件总排名。
