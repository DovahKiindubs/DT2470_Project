# CQT-only 基线代码

当前实现：官方 CQT 的 176 维统计特征，分别训练 Ridge 与 `HistGradientBoostingRegressor`（Histogram Gradient Boosting）。**节奏特征、SSM、有序回归尚未实现。** 原始 CQT、元数据、proposal 均不改动。

## 文件职责

- `features_cqt.py`：读取可信的 `.bin`，校验形状 `(88, T)`、空输入及 NaN/Inf；沿时间提取 88 个均值 + 88 个标准差，形成 176 维全局基线特征。
- `features_cqt_windows.py`：在 5 fps CQT 上使用 50 帧窗口、25 帧步长，以相邻帧正向 CQT-dB 变化均值排序窗口，选最高 20% 窗口；将全局统计、困难窗口统计的均值池化和逐维最大池化拼成 528 维特征。
- `split_data.py`：作曲家名称进行 Unicode/大小写/空白归一化；相同作曲家或共用 YouTube 录音的条目归为同一组，冻结 train/val/test。仍不能代替作曲家别名和曲目身份的人工复核。
- `train.py`：只在训练集拟合 StandardScaler 和 Ridge；在验证集比较 alpha；按 Tau-c 最大、MSE 最小的顺序选参，保存模型和运行记录。
- `train_hgb.py`：复用相同特征和冻结划分，比较 5 个 `max_leaf_nodes` 候选；树模型不需要 StandardScaler，默认只评估验证集。
- `prepare_crnn.py`：只用严格划分中的训练集全部有效帧，计算 88 个 CQT bin 的标准化统计。
- `model_crnn.py`：变长 CQT Dataset、长度分桶、两层时间卷积、双向 GRU、掩码注意力和回归头。
- `train_crnn.py`：支持 CUDA/MPS/CPU，CUDA 默认 AMP、pinned memory、多进程读取、验证集早停和最佳 checkpoint。
- `evaluate.py`：统一计算 Kendall's Tau-c、Spearman、MAE、MSE 和 Acc±1。
- `../tests/test_baseline.py`：特征、受限加载、标签对齐、分组、泄漏防护、指标及命令行端到端测试。

## 运行（终端位于 Project/）

```bash
# 第一次：生成全局和滑窗困难片段特征缓存
python src/features_cqt.py
python src/features_cqt_windows.py

# 全局统计基线
python src/train.py
python src/train_hgb.py

# 滑窗困难片段池化：复用相同严格划分，仅评估验证集
python src/train.py --features datasets/features/cqt_hard_windows.npz --output outputs/cqt_hard_windows/ridge/run_03 --alphas 100 300 1000 3000 10000
python src/train_hgb.py --features datasets/features/cqt_hard_windows.npz --output outputs/cqt_hard_windows/hgb/run_02

# 已存在运行目录时，用新目录避免覆盖
python src/train.py --output outputs/cqt_only/ridge/run_02
python src/train_hgb.py --output outputs/cqt_only/hgb/run_02

# 检查测试和环境
python -W error -m pytest tests -q
python -m pip check
```

特征缓存已在本机生成，**现在不必再次执行前两条特征提取命令**。脚本拒绝覆盖已有特征文件或运行目录；如改变特征算法，使用新的 `--output` 缓存路径，并在训练时用 `--features` 指向它。

默认路径：

- CQT 输入：`datasets/features/cqt/`
- 元数据：`datasets/metadata/new_clean_data.json`
- 全局特征缓存：`datasets/features/cqt_stats.npz`
- 滑窗困难片段特征缓存：`datasets/features/cqt_hard_windows.npz`
- 冻结划分：`datasets/splits/cqt_composer_split.json`
- 全局统计结果：`outputs/cqt_only/`
- 滑窗困难片段结果：`outputs/cqt_hard_windows/`

## 已完成的初步验证

7,901 个输入文件成功生成 176 维有限值特征，没有静默丢弃曲目。固定 seed=42，先留出 20% 的组作为测试集，再取剩余组的 25% 作为验证集。由于分组大小不同，这不是严格的曲目数 60/20/20：实际训练/验证/测试分别为 4,211 / 2,095 / 1,595 条，所有集合均覆盖 0–10 等级。该划分不是作者的官方划分。

在 528 维滑窗困难片段特征上，Ridge 扩展搜索 `[100, 300, 1000, 3000, 10000]` 后仍选中 `alpha=1000`；HGB 在相同候选中仍选中 `max_leaf_nodes=31`。滑窗特征使 Ridge 的验证集 Tau-c 从 `0.5689` 提升到 `0.6433`，Acc±1 从 `51.93%` 提升到 `60.91%`；使 HGB 的 Tau-c 从 `0.6078` 提升到 `0.6534`，Acc±1 从 `56.52%` 提升到 `62.96%`。完整对照见 `outputs/cqt_hard_windows/feature_comparison_validation.json`。

滑窗结果支持“局部 CQT 变化包含额外难度信息”的假设，但仍不是最终测试结论。困难窗口是用谱变化代理定义的，并不等同于音乐学上真实的最难段落；池化仍没有保存窗口之间的完整顺序，也不能替代原始音频的 onset/tempo 特征。

真实测试集尚未评估。只有方法冻结后，才显式执行：

```bash
python src/train.py --evaluate-test --output outputs/cqt_only/ridge/final_run
python src/train_hgb.py --evaluate-test --output outputs/cqt_only/hgb/final_run
```

已有划分文件会优先复用，`--seed` 仅在首次创建划分时生效。不要通过重复换 seed 挑选更好结果。如果后续只下载到一部分音频，节奏对比必须在共同子集上重新跑 CQT 基线，并保持已有分组边界，而不是将全量 CQT 成绩直接与子集节奏成绩比较。

## CRNN 服务器训练

CRNN 直接读取完整 $88\times T$ CQT 序列，不使用 176/528 维缓存。结构为两层 stride-2 `Conv1d`、双向 GRU、掩码注意力和线性回归头。完整 A800 上传、CUDA 环境与命令见根目录 `CRNN_SERVER.md`。

本地已用 CPU 完成 1 epoch smoke run，验证了数据加载、变长 padding/mask、前向反向、验证、早停、最佳 checkpoint 和安全回读。该 smoke run 的分数不作为正式 CRNN 结果。正式 A800 命令默认为：

```bash
python -u src/train_crnn.py --device cuda --batch-size 32 --num-workers 4 \
  --max-epochs 30 --patience 5 --output outputs/cqt_crnn/a800_run_01
```

模型仍使用相同严格 split，并只以 validation Tau-c 选择 epoch；test 默认封存。

## 输出和指标口径

每个运行目录包含 `config.json`、`validation_search.json`、`metrics.json`、`predictions_val.json` 和 `model.joblib`；显式测试时才增加 `predictions_test.json`。Ridge 模型文件包含 StandardScaler；HGB 直接使用原 176 维特征，不应额外拟合标准化器。

- Tau-c / Spearman：使用未舍入的连续模型得分。
- MAE / MSE：使用未截断的等级尺度得分。数值误差按 0–10 编码解释，不表示教学难度等距。
- Acc±1：先截断到 0–10，再 `floor(score + 0.5)` 舍入成等级，与真值相差不超过 1 记为正确。
- 常数真值或常数预测：相关系数记为 JSON `null` 并说明原因，不伪造零相关。
- 同时提供只预测训练集平均等级的基线作为 sanity check。

运行目录不会自动被提交；模型、特征缓存、虚拟环境及逐曲预测已被 Git 忽略。配置、汇总指标、划分和代码可纳入版本管理。本次没有 commit 或 push。

## 安全与限制

`.bin` 来源是 pickle。加载器只允许构造 NumPy 数组所需的白名单类型并限制单文件体积，但仍不应加载不可信 pickle。源哈希记录用于追溯，不代表签名认证。当前实现面向已核验的官方特征文件；不提供任意上传文件的在线服务。

本轮在保留 176 维全局基线的同时，新增了 528 维滑窗困难片段池化，并分别用 Ridge 与 Histogram Gradient Boosting 对照。没有原始音频节奏特征、SSM、Ordinal Logistic Regression，也没有跨多个随机划分的稳定性结论。
