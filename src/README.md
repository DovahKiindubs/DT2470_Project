# CQT-only 基线代码

当前实现：官方 CQT 的 176 维统计特征 + StandardScaler + Ridge。**节奏特征、SSM、有序回归尚未实现。** 原始 CQT、元数据、proposal 均不改动。

## 文件职责

- `features_cqt.py`：读取可信的 `.bin`，校验形状 `(88, T)`、空输入及 NaN/Inf；沿时间提取 88 个均值 + 88 个标准差，保存特征、标签、曲目标识及源文件 SHA-256。输入已经是 dB，不重复取 log/abs、不做逐曲标准化。
- `split_data.py`：作曲家名称进行 Unicode/大小写/空白归一化；相同作曲家或共用 YouTube 录音的条目归为同一组，冻结 train/val/test。仍不能代替作曲家别名和曲目身份的人工复核。
- `train.py`：只在训练集拟合 StandardScaler 和 Ridge；在验证集比较 alpha；按 Tau-c 最大、MSE 最小的顺序选参，保存模型和运行记录。
- `evaluate.py`：统一计算 Kendall's Tau-c、Spearman、MAE、MSE 和 Acc±1。
- `../tests/test_baseline.py`：特征、受限加载、标签对齐、分组、泄漏防护、指标及命令行端到端测试。

## 环境

本机 `.venv` 已安装并验证最小依赖，实际 Python 为 3.14.7。独立的 `requirements-baseline.txt` 记录本次完整版本；原 `requirements.txt` 仍为后续全项目的建议依赖，不是基线锁文件。

另一台机器需要先创建虚拟环境，再安装 `requirements-baseline.txt`。这些精确版本已在本机验证，但未测试其他操作系统。

## 运行（终端位于 Project/）

```bash
# 第一次：从现有 CQT 生成缓存
.venv/bin/python src/features_cqt.py

# 默认运行：生成/复用固定划分，训练并仅评估验证集
.venv/bin/python src/train.py

# 已存在 run_01 时，用新目录避免覆盖
.venv/bin/python src/train.py --output outputs/cqt_only/ridge/run_02

# 检查测试和环境
.venv/bin/python -W error -m pytest tests -q
.venv/bin/python -m pip check
```

特征缓存已在本机生成，**现在不必再次执行第一条**。脚本拒绝覆盖已有特征文件或运行目录；如改变特征算法，使用新的 `--output` 缓存路径，并在训练时用 `--features` 指向它。

默认路径：
- CQT 输入：`datasets/features/cqt/`
- 元数据：`datasets/metadata/new_clean_data.json`
- 特征缓存：`datasets/features/cqt_stats.npz`
- 冻结划分：`datasets/splits/cqt_composer_split.json`
- 第一次真实运行：`outputs/cqt_only/ridge/run_01/`

## 已完成的初步验证

7,901 个输入文件成功生成 176 维有限值特征，没有静默丢弃曲目。固定 seed=42，先留出 20% 的组作为测试集，再取剩余组的 25% 作为验证集。由于分组大小不同，这不是严格的曲目数 60/20/20：实际训练/验证/测试分别为 4,211 / 2,095 / 1,595 条，所有集合均覆盖 0–10 等级。该划分不是作者的官方划分。

在 alpha 候选 `[0.1, 1, 10, 100, 1000]` 中，验证集选中 `alpha=10`。指标以运行目录的 `metrics.json` 为唯一结果来源；这是用来选参的验证集表现，不能称为最终测试成绩或与作者不同协议的成绩直接比较。

真实测试集尚未评估。只有方法冻结后，才显式执行：

```bash
.venv/bin/python src/train.py --evaluate-test --output outputs/cqt_only/ridge/final_run
```

已有划分文件会优先复用，`--seed` 仅在首次创建划分时生效。不要通过重复换 seed 挑选更好结果。如果后续只下载到一部分音频，节奏对比必须在共同子集上重新跑 CQT 基线，并保持已有分组边界，而不是将全量 CQT 成绩直接与子集节奏成绩比较。

## 输出和指标口径

每个运行目录包含 `config.json`、`validation_search.json`、`metrics.json`、`predictions_val.json` 和 `model.joblib`；显式测试时才增加 `predictions_test.json`。模型内包含 StandardScaler，预测时不能再单独拟合标准化器。

- Tau-c / Spearman：使用未舍入的 Ridge 连续得分。
- MAE / MSE：使用未截断的 Ridge 等级尺度得分。数值误差按 0–10 编码解释，不表示教学难度等距。
- Acc±1：先截断到 0–10，再 `floor(score + 0.5)` 舍入成等级，与真值相差不超过 1 记为正确。
- 常数真值或常数预测：相关系数记为 JSON `null` 并说明原因，不伪造零相关。
- 同时提供只预测训练集平均等级的基线作为 sanity check。

运行目录不会自动被提交；模型、特征缓存、虚拟环境及逐曲预测已被 Git 忽略。配置、汇总指标、划分和代码可纳入版本管理。本次没有 commit 或 push。

## 安全与限制

`.bin` 来源是 pickle。加载器只允许构造 NumPy 数组所需的白名单类型并限制单文件体积，但仍不应加载不可信 pickle。源哈希记录用于追溯，不代表签名认证。当前实现面向已核验的官方特征文件；不提供任意上传文件的在线服务。

本轮只完成轻量初步基线：没有节奏特征、SSM、Ordinal Logistic Regression，也没有跨多个随机划分的稳定性结论。
