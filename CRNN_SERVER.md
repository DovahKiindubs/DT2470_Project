# A800 服务器运行 CRNN

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
