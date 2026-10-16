# 服务器重跑：9组共同子集实验

## 本轮范围

仅重跑9组核心实验，全部使用 `datasets/splits/cqt_rhythm_common_v1.json`，train/val/test=4089/2040/1557。seed默认42。HGB四组、early两组、late三组；不再重跑全部历史调参。默认不评估test，不重新提取音频特征，不从旧checkpoint续训。

- HGB：CQT hard-window、加rhythm410、加structure73、加两者。
- early：相同裁剪范围的CQT-only、CQT+rhythm9输入融合。
- late：原生CQT-only分支、rhythm9-only分支、两个分支表示融合。

旧结果和整理前源码完整保存在 `archives/pre_rerun_v1/`。新结果绝不写入归档。分层改动仅调整import和ROOT定位；另让HGB复用已有metadata LF/CRLF等价校验，拒绝真实内容变化。本轮没有改网络层、特征公式、padding、优化器或早停。

## 1. 服务器同步

提交并推送本地整理后代码，再在服务器使用原来成功训练过的Python3.11/PyTorch CUDA `.venv` 执行git拉取。这里没有代你commit/push，也没有连接服务器。

Git当前可能显示旧outputs文件为D、新archives为未跟踪，这是尚未暂存的移动；提交时要把新归档与新源码一并纳入，不能只提交删除项。已有旧数据不会因归档而丢失。

**不要复制macOS的.venv，也不要重新照抄旧requirements freeze安装全部依赖。** 优先复用已成功运行的服务器环境：

```bash
source .venv/bin/activate
python -c "import torch, numpy, scipy, sklearn, joblib; print(torch.__version__); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
python -m pip check
```

git拉取不包含被忽略的CQT/rhythm原序列，服务器必须另有：

- `datasets/features/cqt/*.bin`：共同子集所需CQT原序列。
- `datasets/features/rhythm_windows/*.npz`，包括原 `index.npz`。
- `datasets/features/cqt_hard_windows.npz`、`rhythm_stats.npz`、`structure_stats.npz`。
- `datasets/features/cqt_rhythm_common_v1_normalization.npz`：early两个模式共用。
- `datasets/features/cqt_rhythm_native_normalization.npz`：late三个模式共用。
- `datasets/metadata/new_clean_data.json`。
- `datasets/splits/cqt_rhythm_common_v1.json`及对应`_manifest.json`。

不需要原始audio、176维cqt_stats或旧36维rhythm_temporal。统计缓存覆盖全量/旧音频子集没关系，训练按共同split选取成员，不将额外曲目纳入训练。现有归一化已基于共同train生成，这次可以直接复用，不能替换成全量旧归一化。

配置中的SHA256锁定当前数据版本。普通metadata允许已验证的LF/CRLF差异；split、manifest、NPZ必须保持字节一致，不能为了通过预检删除哈希。建议使用文件传输保存原字节。

**服务器旧输出**：本地归档会通过Git反映已跟踪文件的移动，但服务器自己的未跟踪模型/日志不会被git pull自动搬走。它们不会与 `outputs/rerun_v1/` 冲突；如需归档，先单独核对再移动，不能用 `git clean` 清理。

## 2. 先看命令，再做数据预检

在服务器项目根目录：

```bash
# 默认只打印9组命令；不训练、不建立run目录
python scripts/run_experiments.py --dry-run

# 读取所有所需输入，校验成员、标签、缓存哈希、归一化来源和逐曲序列；不拟合/不评分test
python scripts/run_experiments.py --check
```

预检会读共同子集的序列文件，包括test成员的格式/哈希检查；不以test数据拟合归一化、选择模型或计算成绩。缺文件、坏缓存、成员变化直接报错，不静默减少样本。

## 3. 顺序执行全部9组（推荐）

```bash
python -u scripts/run_experiments.py --execute --device cuda --num-workers 4 --threads 4
```

自动按 HGB4组→early2组→late3组逐个运行，不会同时挤满同一GPU。树模型仍用CPU；CRNN使用CUDA。训练日志直接写入各自的 `train.log`，启动器在终端打印日志路径，无需额外 `tee`。

HGB仍比较5个叶节点候选、300次boosting；CRNN仍使用batch32、最大100轮、patience15、Tau-c累计提升阈值0.001。保留整个运行期间实际最佳checkpoint，不拿最后一轮代替最佳。

只跑某一组：

```bash
python scripts/run_experiments.py --execute --group hgb
python scripts/run_experiments.py --execute --group early
python scripts/run_experiments.py --execute --group late
```

只跑指定实验：

```bash
python scripts/run_experiments.py --execute --experiments late_combined
```

实验名以 `configs/rerun_v1.json` 的id为准，`--dry-run`会列出全部。未知实验名、拼错参数会报错，不会被忽略。

## 4. 输出层级与失败重试

```text
outputs/
  README.md
  rerun_v1/
    hgb/
      cqt/seed_42_run_01/
      cqt_rhythm/seed_42_run_01/
      cqt_structure/seed_42_run_01/
      cqt_rhythm_structure/seed_42_run_01/
    early/
      cqt_only/seed_42_run_01/
      cqt_rhythm/seed_42_run_01/
    late/
      cqt_only/seed_42_run_01/
      rhythm_only/seed_42_run_01/
      combined/seed_42_run_01/
    summaries/
      RESULTS.md
      results.json
```

每次run内：
- `run.json`：唯一运行状态记录、完整命令、seed、输入和源码哈希、开始/结束/异常。
- `train.log`：训练器stdout/stderr，不与其他run混写。
- `artifacts/`：保持原训练器文件格式，包含config、metrics、predictions、model；CRNN还包括history、checkpoint。

运行目录通过原子创建预留，默认拒绝覆盖。同名结果存在时：

```bash
# 跳过已经成功完成且配置、源码和结果仍匹配的run，继续尚未开始的实验
python scripts/run_experiments.py --execute --skip-completed

# 失败实验另起run，不删除旧记录，不是断点续训
python scripts/run_experiments.py --execute --experiments late_combined --run-name run_02
```

`--skip-completed`不会绕过失败/半成品目录；失败则先看日志，修复后用新run名。中断后从头训练，原checkpoint没有完整optimizer/RNG恢复状态。强杀/断电可能留下running标记，先核实进程，不自动把它改成成功。

如以后做多seed，用同一seed清单重复完整对照组，例如 `--seed 43` 会进入 `seed_43_run_01/`；不要只换失败或低分模型的seed，或仅汇报最佳seed。

## 5. 看进度和写报告

```bash
python scripts/run_experiments.py --summarize
```

只扫描这9组新结果，输出 `outputs/rerun_v1/summaries/RESULTS.md`与`results.json`，保留所有attempt和seed，不混入历史归档，也不覆盖outputs/README。每次训练结束也自动刷新。进程运行中可在另一个终端执行汇总；running不代表一定仍有活进程。

一组只在进程退出成功、必要产物齐全、split和seed匹配、逐曲validation预测与指标复算一致后标记completed。失败/中断均保留证据。

所有新实验现在成员相同，但表示和模型仍有差异：HGB是统计向量，early公共裁剪，late原生全长与新预测头；不能把跨组差异全部归因于节奏或融合位置。结构算法和录音同源验证的已有局限没有因重跑消失。

报告最终固定方法后再独立安排test评估；训练启动器不接受`--evaluate-test`，避免重跑时意外反复看test。

## 6. 最终Test-only评估（模型已经冻结后，只运行一次）

当前9组模型已经在validation上完成选择。不要使用训练脚本的`--evaluate-test`，因为那会重新训练。先把包含九组`outputs/rerun_v1/.../seed_42_run_01/`的最新主线同步到服务器，并再次执行数据检查：

```bash
source .venv/bin/activate
python scripts/run_experiments.py --check

# 只打印说明，不访问test
python scripts/evaluate_test.py
```

确认不再根据test修改模型、特征、超参数或选择epoch后，一次性运行：

```bash
python -u scripts/evaluate_test.py \
  --execute \
  --acknowledge-final-test FROZEN_MODELS_FINAL_TEST \
  --device cuda \
  --batch-size 32 \
  --num-workers 4
```

它会：

- 加载当前9个已经保存的`model.joblib`/`model.pt`；
- 核对共同split、validation结果、训练源码哈希、模型哈希、特征和归一化来源；
- 只对冻结test的1557首做推理；不fit、不早停、不选参、不改模型；
- 写入独立目录`outputs/rerun_v1/test_evaluation/final_01/`，包含每组`metrics.json`、`predictions_test.json`以及总表`RESULTS.md`/`results.json`；
- 完成后拒绝用同一目录再次评估，避免反复看test。

正常情况下**不要并行启动第二个test进程**；锁文件会阻止重复执行。若进程因机器故障中断，先检查`release.json`和已经生成的文件；仅在代码、模型和数据完全未变时使用：

```bash
python -u scripts/evaluate_test.py \
  --execute --resume \
  --acknowledge-final-test FROZEN_MODELS_FINAL_TEST \
  --device cuda --batch-size 32 --num-workers 4
```

`--resume`只补未完成的模型，不重新计算已完成项；任何模型、源码、配置或数据变化都会拒绝续跑。不要删除`final_01`后重新评估，也不要根据test结果开`final_02`调参。

Test完成后把`outputs/rerun_v1/test_evaluation/final_01/`提交回Git；报告的逐曲案例仍优先使用validation，test只报告预先冻结的总体结果。Test运行速度主要取决于5个CRNN前向推理，远快于重新训练。

## 7. 本地验证范围

整理前56项训练相关测试通过。完整测试集原先就因本地缺少soundfile无法收集音频预处理测试；本次不改变该音频依赖或重跑音频提取。

最终验证记录见 `reports/rerun_reorganization_verification.json`。合成数据smoke与真实数据预检只验证运行路径，不作为项目实验成绩；正式服务器CUDA训练由你执行。
