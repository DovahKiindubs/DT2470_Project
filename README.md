# 钢琴演奏难度预测

DT2470 Music Informatics 课程项目。项目使用 PSyllabus 钢琴录音数据，根据音频预测 0–10 的有序演奏难度，并研究节奏与结构相关特征能否改善 CQT 基线。

## 方法

- **HGB 回归**：使用固定长度特征，比较 CQT、节奏、结构及其组合。
- **Early CRNN**：将 88 通道 CQT 与 9 通道节奏序列对齐后，在输入端拼接。
- **Late Fusion CRNN**：CQT 与节奏分别编码，再融合高层表示。
- **主要指标**：Kendall’s Tau-c；同时报告 Spearman、MAE、MSE 与 Acc±1。

全部实验使用同一冻结划分：训练 4,089 首、验证 2,040 首、测试 1,557 首，随机种子为 42。最终测试结果中，HGB 的 CQT＋节奏＋结构组合取得最高 Tau-c 0.7204。

## 环境

建议使用 Python 3.12，并在项目根目录创建虚拟环境：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

CUDA 环境中的 PyTorch 请按服务器 CUDA 版本单独安装，避免直接复制其他系统的 `.venv`。`requirements.txt` 是本地开发环境快照，不保证在 Python 3.11 或不同 CUDA 环境中原样安装成功；服务器可按实际 Python/CUDA 版本调整依赖版本。

## 数据准备

数据目录和文件说明见 [`datasets/README.md`](datasets/README.md)。完整复现需要：

- `datasets/features/cqt/*.bin`
- `datasets/features/rhythm_windows/*.npz` 及 `index.npz`
- `datasets/features/cqt_hard_windows.npz`
- `datasets/features/rhythm_stats.npz`
- `datasets/features/structure_stats.npz`
- 两个归一化缓存、metadata、冻结 split 与 manifest

原始音频不是训练最终十组实验的必需输入；只有重新提取节奏或结构特征时才需要。

## 复现实验

所有命令都在项目根目录执行，并显式使用最终配置：

```bash
# 仅打印十组实验命令
python scripts/run_experiments.py --config configs/rerun_v2.json --dry-run

# 校验文件、成员与哈希，不训练
python scripts/run_experiments.py --config configs/rerun_v2.json --check

# 顺序运行全部十组实验
python scripts/run_experiments.py --config configs/rerun_v2.json \
  --execute --device cuda --num-workers 4 --threads 4 \
  --seed 42 --run-name run_01

# 汇总验证集结果
python scripts/run_experiments.py --config configs/rerun_v2.json --summarize
```

只运行某组时可使用 `--group hgb`、`--group early` 或 `--group late`。结果写入 `outputs/rerun_v2/`；同名目录不会被覆盖。

模型全部冻结后，最终测试只运行一次：

```bash
python scripts/evaluate_test.py --config configs/rerun_v2.json \
  --execute --acknowledge-final-test FROZEN_MODELS_FINAL_TEST \
  --device cuda --batch-size 32 --num-workers 4 \
  --seed 42 --run-name run_01
```

测试脚本只加载保存的模型做推理，不重新训练或选参。

## 目录

```text
configs/   最终实验清单
src/       特征、数据、模型、训练与评估实现
scripts/   实验、测试和案例分析入口
tests/     单元测试与合成数据回归测试
datasets/  metadata、划分、特征缓存与本地大文件
outputs/   实验结果（运行后生成）
reports/   报告草稿与图表
```

源码职责见 [`src/README.md`](src/README.md)。
