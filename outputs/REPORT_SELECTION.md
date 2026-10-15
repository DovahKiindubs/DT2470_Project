# 课程报告结果保留清单

## 结论与保留口径

建议报告正文保留 **7 条已有文件的核心结果，分成两组独立消融**；另用一小段保留 early fusion 的两条100轮上限对照（当前只有会话截图，待同步原运行文件）。不是从所有实验挑最高分，而是保留能回答研究问题的完整对照组。

本清单是报告展示建议，不是磁盘清理指令。所有旧运行文件、模型、源码和划分继续保留在原位置；不移动、不删除、不覆盖。路径以下均相对于 `outputs/`。

所有现有成绩都是 validation，而非 test 或官方 MIREX 成绩；不能将 Tau-c 写成分类准确率。Acc±1 为等级误差不超过一级的比例。

## A. 正文主结果一：音乐特征能否补充统计型 CQT？

保留 HGB 这一个模型的完整 CQT／rhythm／structure 组合，不把 Ridge 和全部中间版本一起堆进正文。采用528维 hard-window CQT作为这组基线；rhythm明确为410维全局统计，structure为73维SSM统计，不与后面的9通道序列混称。

验证集 N=2044，配置记载 train=4097、test=1562。以下四个运行的验证曲目集合已逐项核对一致，五项指标已由预测文件复算。

### A1. Hard-window CQT + HGB（必须保留对照）
- 路径：`audio_subset/cqt_hard_windows/hgb/run_01/`
- 输入：528维。
- Tau-c 0.6565；Spearman 0.8011；MAE 1.3779；MSE 3.1241；Acc±1 63.26%。

### A2. Hard-window CQT + rhythm410 + HGB
- 路径：`audio_subset/cqt_hard_windows_rhythm/hgb/run_02/`
- 输入：938维。
- Tau-c 0.6782；Spearman 0.8190；MAE 1.3255；MSE 2.8684；Acc±1 65.41%。

### A3. Hard-window CQT + structure73 + HGB
- 路径：`audio_subset/cqt_hard_windows_structure/hgb/run_01/`
- 输入：601维。
- Tau-c 0.6819；Spearman 0.8213；MAE 1.2817；MSE 2.7543；Acc±1 65.95%。

### A4. Hard-window CQT + rhythm410 + structure73 + HGB
- 路径：`audio_subset/cqt_hard_windows_rhythm_structure/hgb/run_01/`
- 输入：1011维。
- Tau-c 0.7034；Spearman 0.8380；MAE 1.2193；MSE 2.4976；Acc±1 69.37%。

**可以说明**：在已核对的验证曲目上，加入这些人工音乐描述后，所选模型的指标更高。

**必须披露**：A1记录的split SHA为 `146569a69c55975d981614481b2ccd8ac64279e890bc3fc62571a83165af7bd7`，A2–A4为 `9498468f276ac7b54384e1893ee5dff8a12276c7a2efc645cf5fdecc1b2c383c`。A2–A4的哈希已对应当前split的CRLF版本；A1的旧哈希尚未在本地Git中的split版本里复现。验证成员相同不等于训练成员已被证明相同，不能直接把A1哈希差异称作纯换行问题。正式提交前应核验旧split或以固定split重跑A1；未补证则标注这组为探索性结果，不称严格受控消融。

此外，现有structure是算法估计的SSM统计，并非人工标注的真实段落；novelty实现与标准Foote方法不同，报告应按实际代码描述，不能宣称已证明音乐结构或演奏难度机制。

## B. 正文主结果二：节奏独立编码后是否有增益？

保留 **同一 late-fusion 实现中的全部三组**，不只保留combined。验证集 N=2040，train=4089、test=1557；相同冻结split、原生输入预处理、归一化缓存、训练预算和早停规则，seed=42。

### B1. 新版 CQT-only 分支
- 路径：`cqt_rhythm_common_v1/late_fusion/cqt_only/run_01/`
- Tau-c 0.6451；Spearman 0.7887；MAE 1.4133；MSE 3.3092；Acc±1 63.28%。
- 最佳epoch 61；运行76轮后早停。

### B2. Rhythm-only 分支
- 路径：`cqt_rhythm_common_v1/late_fusion/rhythm_only/run_01/`
- 输入：9通道局部节奏序列。
- Tau-c 0.6331；Spearman 0.7742；MAE 1.4649；MSE 3.4571；Acc±1 59.07%。
- 最佳epoch 61；运行76轮后早停。

### B3. CQT + rhythm Late Fusion
- 路径：`cqt_rhythm_common_v1/late_fusion/combined/run_01/`
- 独立编码、各自投影和LayerNorm、拼接后学习MLP，无固定模态权重。
- Tau-c 0.6841；Spearman 0.8237；MAE 1.2544；MSE 2.7423；Acc±1 68.63%。
- 最佳epoch 71；运行86轮后早停。

**可以说明**：本次验证实验中，双分支优于对应的两个单分支，节奏单独也具有预测信号。

**不能说明**：late fusion全面超过所有旧模型、增益具有统计显著性或来自单一音乐机制。双分支参数更多，尚缺多seed稳定性证据；新CQT-only的预测头、表示压缩和预处理与旧CRNN不同，不能隐藏其性能下降。

## C. 必须保留的负结果与局限（简短讨论，不挤占主表）

### C1–C2. Early fusion，100轮上限的成对实验

会话截图记录：同一N=2040验证集，旧架构CQT-only Tau-c约0.6957，CQT+rhythm early fusion约0.6891。应保留这组“没有观察到提升”的结果，而非只写late fusion成功。

预期路径（本次本地未找到完整运行文件）：
- `cqt_rhythm_common_v1/cqt_only/run_100e_01/`
- `cqt_rhythm_common_v1/cqt_rhythm/run_100e_01/`

状态：**用户截图已报告完成，原始结果待同步**，不能填成“未做过”，也不能与已核验文件同级使用。请同步config、metrics、history、predictions和checkpoint后核验；在此之前不把截图数值列入本地可复算主表。

本地存在的 `cqt_rhythm_common_v1/cqt_rhythm/run_01/` 是30轮上限版本（Tau-c 0.6780），不能冒充100轮版本，也没有匹配的30轮CQT-only结果可作正式成对比较。

旧CQT-only截图排序优于B3，是重要局限，不能因为不支持新方案就删除。Early与Late之间同时改变了输入范围、归一化、预测头和模型容量，因此不能把差异全部归因于融合位置。

## D. 补充材料：最多展示一个小对照组

若篇幅允许，选同一全量严格split（N=2095）的3条结果解释项目演进：
- `cqt_only/ridge/run_01/`：176维全局CQT + Ridge。
- `cqt_only/hgb/run_01/`：176维全局CQT + HGB。
- `cqt_hard_windows/hgb/run_01/`：528维局部难点池化 + HGB。

它们回答“为什么从全局统计转向局部/时序表示”，不与A/B的不同子集直接排名。旧 `cqt_crnn/a800_run_01/`（N=2095）保留在历史记录；需要时仅作背景参考。

## E. 历史归档：不必在正文逐项展开，但原文件全留

- 10维rhythm初版、36维池化rhythm、中间410维组合探索。
- 完整Ridge组合、单独structure和其他早期探索。
- Ridge不同alpha搜索的重复run、30轮early-fusion试运行。
- 模型checkpoint、源特征哈希、原split和旧代码。

“归档”在这里仅表示不进入正文，不执行目录搬迁或删除。不要只保留最高分；保留失败和改进轨迹用于解释结论。

## F. 报告图与最终交付

建议图只做三类：
1. 模型/特征流程图，清楚区分人工410维节奏与9通道节奏序列。
2. B组的三条验证Tau-c训练曲线，标记最佳轮与早停，不用最后一轮替代最佳模型。
3. B1/B3的等级误差或典型误排曲目对照；若展示SSM实例，必须先回听/核验，不能把待做的音乐实例验证写成已完成。

保存每个主结果的config、metrics、predictions_val、history（如有）、模型和对应输入/划分来源；报告正文写关键指标，不复制所有日志。

最终test安排：先确定需要汇报的方法与对照，再按各自预先冻结的测试集统一评估；不按test选特征或选最好seed。若没有执行，只能明确写validation结果，不能把validation改名test。本次没有新训练或test评分。

## G. 当前进度

- A组4条：文件与验证预测已核对；A1训练split旧哈希待补证。
- B组3条：结果已同步、指标复算、同组来源/训练设置核对完成。
- C组成对负结果：截图已知，完整文件待同步。
- 多seed稳定性与最终test：尚未完成。
- 音乐结构实例验证：仅讨论过方案，尚未完成。

本文件是报告选材入口；`EXPERIMENT_LOG.md`保留研究过程，`README.md`为全量结果导航。以后新增结果不能仅因分数更高就替代完整对照组。
