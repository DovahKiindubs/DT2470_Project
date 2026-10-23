# 源码导航

源码按“特征 → 数据 → 模型 → 训练 → 实验调度”分层。根层的部分同名脚本是兼容入口，实际实现位于子目录中，请勿随意删除。

## 目录职责

- `features/`
  - `cqt.py`：CQT读取与全局统计。
  - `cqt_windows.py`：滑窗及高活动窗口池化。
  - `rhythm.py`：410维节奏汇总特征。
  - `rhythm_windows.py`：9通道节奏时序特征。
  - `structure.py`：基于chroma自相似矩阵的73维结构相关统计。
- `data/`
  - metadata、划分和哈希校验。
  - HGB特征拼接。
  - Early Fusion时间对齐与Late Fusion双输入数据集。
  - 训练集归一化与音频预处理。
- `models/`
  - `model_crnn.py`：Conv1D、BiGRU、masked attention与回归头。
  - `model_late_fusion.py`：CQT和节奏独立分支及表示融合。
- `training/`
  - `hgb.py`：Histogram-based Gradient Boosting回归。
  - `crnn.py`：Early Fusion及单输入CRNN训练。
  - `late_fusion.py`：双分支Late Fusion训练。
  - `ridge.py`：早期线性基线，保留用于追溯。
- `reporting/metrics.py`：Tau-c、Spearman、MAE、MSE与Acc±1。
- `experiments/suite.py`：十组实验调度、输入预检、状态记录和汇总。
- `experiments/evaluate_test.py`：冻结模型的最终test-only推理。

## 推荐入口

在项目根目录执行：

```bash
python scripts/run_experiments.py --config configs/rerun_v2.json --dry-run
python scripts/run_experiments.py --config configs/rerun_v2.json --check
python scripts/run_experiments.py --config configs/rerun_v2.json --execute --group hgb
python scripts/run_experiments.py --config configs/rerun_v2.json --execute --group early
python scripts/run_experiments.py --config configs/rerun_v2.json --execute --group late
python scripts/run_experiments.py --config configs/rerun_v2.json --summarize
```

直接运行分层模块时，使用：

```bash
PYTHONPATH=src python -m training.crnn --help
PYTHONPATH=src python -m training.late_fusion --help
PYTHONPATH=src python -m training.hgb --help
```

## 兼容入口

以下根层文件转发到分层实现，用于兼容旧测试和历史命令：

```text
train.py / train_hgb.py / train_crnn.py / train_late_fusion.py
features_cqt.py / features_cqt_windows.py / features_rhythm.py
features_rhythm_temporal.py / features_structure.py
prepare_crnn.py / prepare_late_fusion.py / preprocess_audio.py
summarize_experiments.py
```

## 实验边界

- 最终配置为 `configs/rerun_v2.json`，包含HGB 4组、Early 3组和Late 3组。
- 全部实验共享冻结共同子集，随机种子为42。
- HGB使用固定长度CQT、节奏和结构统计。
- CRNN使用保留时间顺序的CQT与节奏序列；当前结构特征是整曲73维汇总，因此未直接送入CRNN。
- 默认训练入口只评估validation；test通过独立脚本在模型冻结后运行。
- 跨Early与Late的性能差异还包含输入时间网格、分支宽度和预测头差异，不能全部归因于融合位置。
