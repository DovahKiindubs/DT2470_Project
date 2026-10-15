# 源码导航

本轮只做目录分层、必要import/ROOT路径调整和新增实验调度；模型、特征计算、损失函数、优化器与早停算法保持原样。整理前源码快照在 `../archives/pre_rerun_v1/source_snapshot/src/`。

## 实际实现按职责查找

- `features/`：`cqt.py`全局统计、`cqt_windows.py`困难窗池化、`rhythm.py`410维节奏统计、`rhythm_windows.py`9通道窗口序列、`structure.py`SSM统计。
- `data/`：音频读取、冻结划分校验、统计特征拼接、early时间对齐与native两输入数据集；`prepare_crnn.py`/`prepare_late_fusion.py`训练集归一化；`preprocess_audio.py`旧音频QC。
- `models/`：`model_crnn.py`原CRNN和原有Dataset/Sampler；`model_late_fusion.py`独立双分支。为避免改变逻辑，本次不继续拆分这些类。
- `training/`：`ridge.py`、`hgb.py`、`crnn.py`、`late_fusion.py`，保存原训练算法及CLI参数。
- `reporting/metrics.py`：Tau-c、Spearman、MAE、MSE、Acc±1。
- `reporting/legacy_registry.py`：旧通用索引生成器，保留用于历史追溯；当前重跑不要用它覆盖目录导航。
- `experiments/suite.py`：9组统一入口、输入预检、唯一运行目录、日志、状态与汇总。不重复实现模型训练。

## 入口与兼容

推荐在工程根目录：

```bash
python scripts/run_experiments.py --dry-run
python scripts/run_experiments.py --check
python scripts/run_experiments.py --execute --group hgb
python scripts/run_experiments.py --execute --group early
python scripts/run_experiments.py --execute --group late
python scripts/run_experiments.py --summarize
```

根层旧的 `train_crnn.py`、`features_rhythm.py` 等文件现在只是兼容入口，旧import和 `python src/train_crnn.py ...` 仍可使用；实际实现只维护分层目录里的那一份。不要删除这些入口，否则旧测试/成员命令会失效。

要直接运行分层模块：`PYTHONPATH=src python -m training.crnn --help`，不是直接 `python src/training/crnn.py`。正式重跑优先使用统一入口，其配置显式锁定共同子集与输出目录；旧入口默认路径保留原行为，不适合新的9组套件。

## 实验边界

- 全部重跑共享 `cqt_rhythm_common_v1.json`，不重新随机切分；train/val/test=4089/2040/1557。
- HGB使用528维CQT池化、可选410维节奏/73维结构缓存，不使用原始序列。
- early两组都使用同一公共时间裁剪，包含CQT-only对照；late三组都保留原生序列。
- 统一成员不代表统一网络或预处理；early与late之间不能将所有差异归因于融合位置。
- 默认不评分test；不会在本地重跑真实训练；无需重新下载或提取audio。
- 原结构novelty算法、节奏confidence定义和padding行为均未改，已有音乐学/方法限制仍存在。

完整服务器命令、数据清单和失败重试方式见 `../RERUN_SERVER.md`。
