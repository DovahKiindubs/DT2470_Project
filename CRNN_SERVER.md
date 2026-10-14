# CQT＋rhythm CRNN：共同子集 v1 服务器运行

## 当前入口（优先使用本节）

代码通过 Git 同步，不需要打包，也不要复制 macOS 的 `.venv`。使用服务器上已经成功跑过 CQT CRNN 的 Python 3.11 / PyTorch CUDA 环境；本次新增融合无需 librosa、soundfile 或原始 audio，不增加训练依赖。

### 需要同步的文件

- 代码：`src/data/aligned_sequences.py`、`src/models/model_crnn.py`、`src/prepare_crnn.py`、`src/train_crnn.py`，以及原有其余 src 模块。
- 测试：`tests/test_crnn_rhythm.py`，保留 `tests/test_baseline.py`。
- 冻结清单：`datasets/splits/cqt_rhythm_common_v1.json` 和 `datasets/splits/cqt_rhythm_common_v1_manifest.json`。
- 元数据：`datasets/metadata/new_clean_data.json`。
- 数据：`datasets/features/cqt/*.bin` 与 `datasets/features/rhythm_windows/*.npz`（**包括原 index.npz**）。至少保留冻结清单纳入的 7686 对文件；多传其他曲目不会进入实验。
- 新标准化：`datasets/features/cqt_rhythm_common_v1_normalization.npz`，已在本地生成，可以传服务器，也可按下方命令在服务器重建一次。不要替换旧 `cqt_train_normalization.npz`。

注意 `rhythm_windows/` 被 Git 忽略，单纯 git pull 不会带来这些数据；需要另外传输。不要重写 index，清单与每曲缓存的哈希会在加载时校验。未在本次执行 git commit/push 或服务器操作。

### 1. 检查环境和代码

在服务器仓库根目录执行：

```bash
source .venv/bin/activate
python -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
python -m pip check
python -W error -m pytest tests/test_baseline.py tests/test_crnn_rhythm.py -q
```

不要重新照抄本文后半部分的旧 Python 3.12 安装方式，也不要使用 `torch --no-deps`。若当前 CUDA 环境异常，先解决具体报错再训练。

### 2. 生成新的共同归一化（只需一次）

本地已生成该文件。**已传过去则跳过这一步；文件存在时命令会拒绝覆盖。** 如需试不同时间参数，另用新的缓存名并在训练命令中保持参数一致。

```bash
python -u src/prepare_crnn.py \
  --split datasets/splits/cqt_rhythm_common_v1.json \
  --rhythm-dir datasets/features/rhythm_windows \
  --output datasets/features/cqt_rhythm_common_v1_normalization.npz
```

两种模式共用这个 97 通道缓存。CQT-only 仅取前 88 通道，但保留与融合模式完全相同的时间裁剪。统计仅使用冻结 train 的 4089 首；逐通道排除无效观测，不使用 val/test 分布拟合。缓存保存 split、train ids、特征内容签名和时间映射参数；不匹配时报错，不自动跳过。

### 3. 先跑公平的 CQT-only 对照

**这里也必须保留 `--rhythm-dir`**，用于相同时间范围的裁剪；不传它就是旧的全长输入路径。

```bash
set -o pipefail
python -u src/train_crnn.py \
  --feature-mode cqt \
  --rhythm-dir datasets/features/rhythm_windows \
  --split datasets/splits/cqt_rhythm_common_v1.json \
  --normalization datasets/features/cqt_rhythm_common_v1_normalization.npz \
  --device cuda --batch-size 32 --num-workers 4 \
  --max-epochs 30 --patience 5 --seed 42 \
  --output outputs/cqt_rhythm_common_v1/cqt_only/run_01 \
  2>&1 | tee cqt_common_v1_run_01.log
```

### 4. 跑 CQT＋rhythm 融合

```bash
set -o pipefail
python -u src/train_crnn.py \
  --feature-mode cqt-rhythm \
  --rhythm-dir datasets/features/rhythm_windows \
  --split datasets/splits/cqt_rhythm_common_v1.json \
  --normalization datasets/features/cqt_rhythm_common_v1_normalization.npz \
  --device cuda --batch-size 32 --num-workers 4 \
  --max-epochs 30 --patience 5 --seed 42 \
  --output outputs/cqt_rhythm_common_v1/cqt_rhythm/run_01 \
  2>&1 | tee cqt_rhythm_common_v1_run_01.log
```

输出目录已存在时会拒绝覆盖；新实验使用 run_02 等新目录以及新日志名。两组都默认不评估 test，以 validation Tau-c 选 epoch、MSE 并列决胜。比较的是当前验证集 2040 首，不是旧全量 2095 首。均从头训练，不加载旧 CQT checkpoint。初始化输入层大小不同，不承诺两组随机权重逐元素相同；后续可用相同 seed 列表进行重复实验。

### 本次实现细节与边界

- `--feature-mode cqt-rhythm`：88维CQT＋9维节奏，沿通道拼接为97×T；卷积输入通道可变，其余两层 Conv1d、BiGRU、masked attention、MSE、优化器和默认超参不变。
- 完整节奏窗口按 `times + win_sec/2` 定位到中心，在 CQT 原始网格上做线性插值；只保留两侧覆盖的交集，不外推、不把整曲拉伸为相同长度。
- 当前默认 `--cqt-fps 5 --cqt-time-offset 0`，代表**名义 5fps、两种特征起点相同的假设**。官方生成脚本先按原始高帧率截取再 Fourier 重采样、帧数取整；缓存没有完整重采样时间元信息，不能宣称这里是恢复后的精确时间轴。来源： https://raw.githubusercontent.com/PRamoneda/audio-difficulty/main/get_cqt.py 。若核实存在系统性偏移，必须在准备与训练使用相同的明确参数并重建新标准化；不能靠调偏移追求验证成绩。
- 首尾窗口不规则步长按实际 `times` 处理；若只有一窗且没有 `window_end_times`，报错而非猜测短曲窗中心。冻结 v1 没有这类单窗样本。
- IOI mean > 0 时 IOI 三通道有效；tempo要求有起音、非静音且至少能计算IOI；confidence只在有起音且非静音时有效。线性插值两端任一无效则该目标值无效；精确落在有效窗口中心不受相邻无效窗影响。
- 无效值不进入训练均值/std；标准化后填0（即训练均值填补），**不追加mask特征通道**，保持97维。它不是“真实IOI为零”，模型也没有显式缺失指示器。confidence仍是tempogram峰均比，不是校准过的BPM置信度。
- NFC匹配解决三组文件名编码差异，不重命名源文件；哈希检查仅接受已核实的metadata LF/CRLF变体，不忽略真实数据变化。
- 每曲读取校验内容哈希并执行同一对齐函数，避免旧缓存混用；初始扫描与每epoch磁盘加载有开销。
- 17个疑似录音不匹配样本仍排除，198个缺节奏缓存样本仍排除；冻结划分没有改变。**波形同源和精确录音对齐仍未验证。**
- 原卷积padding/BatchNorm问题本次未修，GRU/attention掩码保持原状；不把它与新增节奏的影响混在本次改动里。
- 默认不访问test的数值做标准化/训练/评分；初始化仅检查完整冻结成员的文件名与索引/manifest。运行 `--evaluate-test` 仍为从头训练后测试，不是加载已有模型直接评估，调参阶段不要开启。

---

# 历史 CQT-only 运行记录（以下不适用于本次融合）

保留旧记录供追溯；安装路径、依赖文件及基线数字可能已过时。新实验请只使用上方当前入口。

目标环境：Ubuntu、NVIDIA A800-SXM4-40GB、驱动显示 CUDA 12.9、项目级 Python venv。

## 1. 上传内容

将整个 `Project/` 上传服务器最简单，但可排除本地产物：

```bash
rsync -av --progress \
  --exclude '.venv' --exclude '.git' --exclude 'outputs' \
  "/local/path/Project/" user@server:/server/path/Project/
```

CRNN 必须有以下数据：

```text
datasets/features/cqt/*.bin                 # 7,901 个 CQT 序列
datasets/metadata/new_clean_data.json
datasets/splits/cqt_composer_split.json
datasets/features/cqt_train_normalization.npz
```

`cqt_train_normalization.npz` 已由严格划分中的 4,211 首训练曲目计算；若划分或元数据改变，必须删除旧文件后重新运行 `prepare_crnn.py`。

## 2. 创建环境

推荐 Python 3.12；不要复制 macOS 的 `.venv` 到 Linux。

```bash
cd /server/path/Project
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-server.txt
# 安装官方 CUDA wheel；A800 驱动 12.9 可运行较低 CUDA runtime 的 wheel。
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
python -m pip check
```

服务器驱动的 CUDA 12.9 不要求本机安装与 PyTorch 完全相同的 toolkit；关键是 PyTorch wheel 自带的 CUDA runtime 能被驱动支持。安装后先检查：

```bash
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('torch CUDA runtime:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

如果 `CUDA available` 是 `False`，不要开始训练；重新安装 PyTorch 官方 CUDA wheel。不要用 CPU 结果冒充 A800 训练结果。

## 3. 运行测试

```bash
python -W error -m pytest tests -q
```

本地已通过 26 项测试，包括 CRNN 张量形状、变长掩码、注意力归一化和反向传播。

## 4. 准备训练集标准化

已上传 `cqt_train_normalization.npz` 时可跳过。否则执行：

```bash
python src/prepare_crnn.py
```

统计只使用 train 集全部有效帧，不读取 validation/test 分布。

## 5. 正式训练

A800 40GB 建议从 batch size 32 开始：

```bash
python -u src/train_crnn.py \
  --device cuda \
  --batch-size 32 \
  --num-workers 4 \
  --max-epochs 30 \
  --patience 5 \
  --output outputs/cqt_crnn/a800_run_01 \
  2>&1 | tee crnn_a800_run_01.log
```

默认启用 CUDA AMP、pinned memory、长度分桶、梯度裁剪和验证集早停。若显存仍很充裕，可以新建一次实验把 `--batch-size` 调至 48 或 64；不要在同一运行目录覆盖结果。若 OOM，先降至 16。

代码每当验证集表现刷新时写入 `best_checkpoint.pt`，训练结束后另存 `model.pt`、`config.json`、`history.json`、`metrics.json` 和验证集逐曲预测。当前 test 尚未评估，默认运行也**不评估 test**。

实时监控：

```bash
watch -n 1 nvidia-smi
```

## 6. 结果判读

主要看：

```text
outputs/cqt_crnn/a800_run_01/metrics.json
outputs/cqt_crnn/a800_run_01/history.json
```

验证集参考结果：

- 全局统计 HGB：Tau-c 0.6078；
- 滑窗困难片段 HGB：Tau-c 0.6534。

CRNN 应首先与滑窗 HGB 的验证结果比较。若 CRNN 未超过它，也仍是有价值的结论：当前数据量、5 fps CQT 或训练目标可能不足以让序列模型优于人工池化。

只有架构与训练方案冻结后才运行 test：

```bash
python -u src/train_crnn.py \
  --device cuda --batch-size 32 --num-workers 4 \
  --max-epochs 30 --patience 5 --evaluate-test \
  --output outputs/cqt_crnn/a800_final_test
```

注意：当前脚本会重新训练后再测试，而不是拿已有验证模型直接开 test。正式 test 只能做一次，并在报告中说明模型选择规则。

## 7. 当前模型结构与限制

```text
CQT 88×T
→ Conv1d(stride 2) × 2，时间缩短约 4 倍
→ Bidirectional GRU
→ masked attention pooling
→ linear regression score
```

损失为 MSE，模型选择以 validation Tau-c 为主、MSE 为并列决胜。模型直接处理完整 CQT 序列并屏蔽 padding，但输入仍只有约 5 fps，因此无法恢复原始音频中的精细 onset 时序。本轮只尝试一个轻量架构，不能把单次结果解释为 CRNN 的性能上限。
