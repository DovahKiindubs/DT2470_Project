# 实验结果导航

## 写报告先看这里

- [REPORT_SELECTION.md](REPORT_SELECTION.md)：正文保留的7条核心结果、early fusion负结果、补充材料与历史归档规则。
- [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)：E01–E06研究过程、当前结论和待补证据。
- 原始run目录仍是配置、预测、模型与训练记录的权威来源；本次只整理文档，不移动、删除或覆盖任何run，也不改源码。

## 当前本地盘点

- 带 `metrics.json` 的运行：**38**（不等于所有潜在临时/未完成目录）。
- 标记已评估test的运行：**0**。
- 全量严格划分，validation N=2095：**6**。
- 音频子集，validation N=2044：**28**。
- 冻结共同子集，validation N=2040：**4**（early30轮1条＋late三组）。
- 100轮上限early/CQT-only对照目前仅有会话截图，未发现对应完整run，不计入上述38条。

## 比较规则

N相同不等于同一划分；应核对实际成员、训练数据、预处理、模型差异和调参预算。哈希不同应检查具体内容，不能一概解释为换行，也不能仅凭哈希就否定成员等价。A组旧HGB基线的训练split哈希尚待补证，见选材清单。不同N的结果分开报告，不做总榜；所有当前成绩均为validation。

## Frozen common-v1 CRNN family (N=2040)

| Run | N | Tau-c | Spearman | MAE | MSE | Acc±1 | Test |
|---|---:|---:|---:|---:|---:|---:|---|
| `cqt_rhythm_common_v1/late_fusion/cqt_only/run_01` | 2040 | 0.6451 | 0.7887 | 1.4133 | 3.3092 | 0.6328 | no |
| `cqt_rhythm_common_v1/late_fusion/rhythm_only/run_01` | 2040 | 0.6331 | 0.7742 | 1.4649 | 3.4571 | 0.5907 | no |
| `cqt_rhythm_common_v1/late_fusion/combined/run_01` | 2040 | 0.6841 | 0.8237 | 1.2544 | 2.7423 | 0.6863 | no |
| `cqt_rhythm_common_v1/cqt_rhythm/run_01`（early30，历史参考） | 2040 | 0.6780 | 0.8167 | 1.3061 | 2.8293 | 0.6534 | no |

上面前三条属于同一新版架构族的消融；early30不是同预处理/同预测头的单变量融合位置对照。

## Full strict split family (N=2095)

| Run | N | Tau-c | Spearman | MAE | MSE | Acc±1 | Test |
|---|---:|---:|---:|---:|---:|---:|---|
| `cqt_crnn/a800_run_01` | 2095 | 0.6806 | 0.8206 | 1.3800 | 3.1493 | 0.6353 | no |
| `cqt_hard_windows/hgb/run_01` | 2095 | 0.6534 | 0.7974 | 1.3802 | 3.1411 | 0.6296 | no |
| `cqt_hard_windows/ridge/run_01` | 2095 | 0.6433 | 0.7908 | 1.4580 | 3.4565 | 0.6091 | no |
| `cqt_hard_windows/ridge/run_02` | 2095 | 0.6433 | 0.7908 | 1.4580 | 3.4565 | 0.6091 | no |
| `cqt_only/hgb/run_01` | 2095 | 0.6078 | 0.7579 | 1.5444 | 3.7181 | 0.5652 | no |
| `cqt_only/ridge/run_01` | 2095 | 0.5689 | 0.7236 | 1.6445 | 4.1732 | 0.5193 | no |

## Audio subset feature ablations (N=2044)

| Run | N | Tau-c | Spearman | MAE | MSE | Acc±1 | Test |
|---|---:|---:|---:|---:|---:|---:|---|
| `audio_subset/cqt_hard_windows_rhythm_structure/hgb/run_01` | 2044 | 0.7034 | 0.8380 | 1.2193 | 2.4976 | 0.6937 | no |
| `audio_subset/cqt_hard_windows_rhythm_temporal_structure/hgb/run_01` | 2044 | 0.7013 | 0.8357 | 1.2287 | 2.5249 | 0.6957 | no |
| `audio_subset/cqt_hard_windows_rhythm_temporal_structure/ridge/run_01` | 2044 | 0.6997 | 0.8367 | 1.2611 | 2.6614 | 0.6590 | no |
| `audio_subset/cqt_hard_windows_rhythm_structure/ridge/run_01` | 2044 | 0.6962 | 0.8342 | 1.2712 | 2.6660 | 0.6619 | no |
| `audio_subset/cqt_hard_windows_rhythm_temporal/hgb/run_01` | 2044 | 0.6875 | 0.8271 | 1.2842 | 2.7204 | 0.6575 | no |
| `audio_subset/cqt_hard_windows_structure/ridge/run_01` | 2044 | 0.6858 | 0.8260 | 1.3126 | 2.8382 | 0.6551 | no |
| `audio_subset/cqt_hard_windows_structure/hgb/run_01` | 2044 | 0.6819 | 0.8213 | 1.2817 | 2.7543 | 0.6595 | no |
| `audio_subset/cqt_hard_windows_rhythm/hgb/run_02` | 2044 | 0.6782 | 0.8190 | 1.3255 | 2.8684 | 0.6541 | no |
| `audio_subset/cqt_hard_windows_rhythm_temporal/ridge/run_01` | 2044 | 0.6766 | 0.8175 | 1.3473 | 3.0116 | 0.6394 | no |
| `audio_subset/cqt_hard_windows_rhythm/hgb/run_01` | 2044 | 0.6692 | 0.8124 | 1.3481 | 2.9786 | 0.6482 | no |
| `audio_subset/rhythm_structure/hgb/run_01` | 2044 | 0.6690 | 0.8066 | 1.3126 | 2.9211 | 0.6614 | no |
| `audio_subset/cqt_hard_windows_rhythm/ridge/run_02` | 2044 | 0.6669 | 0.8093 | 1.3616 | 3.0687 | 0.6385 | no |
| `audio_subset/cqt_hard_windows_rhythm/ridge/run_01` | 2044 | 0.6617 | 0.8061 | 1.4002 | 3.1955 | 0.6199 | no |
| `audio_subset/cqt_hard_windows/hgb/run_01` | 2044 | 0.6565 | 0.8011 | 1.3779 | 3.1241 | 0.6326 | no |
| `audio_subset/cqt_rhythm/hgb/run_02` | 2044 | 0.6506 | 0.7960 | 1.4044 | 3.1876 | 0.6130 | no |
| `audio_subset/rhythm_structure/ridge/run_01` | 2044 | 0.6473 | 0.7888 | 1.4500 | 3.3594 | 0.5988 | no |
| `audio_subset/cqt_hard_windows/ridge/run_01` | 2044 | 0.6472 | 0.7942 | 1.4537 | 3.4075 | 0.6047 | no |
| `audio_subset/cqt_rhythm/ridge/run_02` | 2044 | 0.6411 | 0.7870 | 1.4438 | 3.4073 | 0.6208 | no |
| `audio_subset/rhythm_temporal/hgb/run_01` | 2044 | 0.6374 | 0.7797 | 1.4466 | 3.3691 | 0.6042 | no |
| `audio_subset/cqt_rhythm/hgb/run_01` | 2044 | 0.6344 | 0.7805 | 1.4521 | 3.4119 | 0.5900 | no |
| `audio_subset/structure/hgb/run_01` | 2044 | 0.6271 | 0.7683 | 1.4510 | 3.4718 | 0.6208 | no |
| `audio_subset/structure/ridge/run_01` | 2044 | 0.6254 | 0.7682 | 1.4903 | 3.6374 | 0.5915 | no |
| `audio_subset/cqt_rhythm/ridge/run_01` | 2044 | 0.6146 | 0.7640 | 1.5336 | 3.6988 | 0.5651 | no |
| `audio_subset/cqt_only/hgb/run_01` | 2044 | 0.6068 | 0.7572 | 1.5549 | 3.7612 | 0.5543 | no |
| `audio_subset/rhythm_temporal/ridge/run_01` | 2044 | 0.5977 | 0.7444 | 1.6209 | 4.3395 | 0.5377 | no |
| `audio_subset/rhythm/hgb/run_01` | 2044 | 0.5866 | 0.7332 | 1.5727 | 3.9228 | 0.5489 | no |
| `audio_subset/cqt_only/ridge/run_01` | 2044 | 0.5706 | 0.7249 | 1.6451 | 4.1731 | 0.5166 | no |
| `audio_subset/rhythm/ridge/run_01` | 2044 | 0.5382 | 0.6805 | 1.7484 | 4.8830 | 0.4980 | no |

## 索引与刷新注意

- [REPORT_SELECTION.md](REPORT_SELECTION.md)：当前人工筛选的报告入口。
- [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)：当前实验过程记录。
- `summaries/experiment_index.json` 与 `summaries/experiment_index.csv`：**历史35条索引快照，尚未包含late三组，本轮未改写，不作为最新完整清单。**
- `summaries/audio_subset/feature_comparison_validation.json`：历史28条音频子集结果映射。
- 本页保留全量38条导航；原始run文件是事实来源。

旧 `python src/summarize_experiments.py` 会重新生成README与机器索引，并覆盖本页的人工导航。当前未修改该脚本；不要将它当作无损更新报告清单的命令。以后刷新应保留本页报告入口、分组限制和待补证据，`REPORT_SELECTION.md`不会由旧脚本自动维护。

本页为late三组同步后的人工整理版；仅更新导航与说明，训练代码、原始run、数据划分均未改动。
