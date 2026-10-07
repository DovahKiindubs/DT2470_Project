# Piano Performance Difficulty Prediction

DT2470 Music Informatics 三人课程项目。

研究目标、范围和评估要求以 [final_project_proposal.md](final_project_proposal.md) 为准。该文件保持原样；本 README 只约定目录与协作方式，不新增研究任务。

## 当前状态

用户已放入 PSyllabus 发布文件。现已归档原始 ZIP、整理元数据，并解压及逐文件校验 7,901 份 CQT；CQT 文件名与元数据键完全对应。详情见 [datasets/README.md](datasets/README.md) 和 [校验清单](datasets/inventory.json)。**原始音频尚未放入；官方划分不满足 composer-disjoint，仍需生成项目划分。** 尚未安装环境、实现模型或训练。

## 目录

```text
Project/
├── final_project_proposal.md  # 最终 proposal，原样保留
├── README.md                  # 项目入口与协作说明
├── requirements.txt           # 建议依赖，尚未锁定验证
├── .gitignore                 # 排除数据、大文件、环境和密钥
├── datasets/
│   ├── README.md              # 数据位置、格式与使用注意事项
│   ├── inventory.json         # 本次校验清单与官方划分检查
│   ├── archives/              # 保留原始 ZIP，不上传 Git
│   ├── audio/                 # 原始音频，目前未放入
│   ├── cqt/                   # 解压后的 CQT .bin 文件，本地保留
│   ├── metadata/              # 原始标签、元数据、官方划分，本地保留
│   ├── splits/                # 待生成本项目的作曲家隔离划分
│   └── features/              # 待生成的固定长度特征缓存
├── src/                       # 可复用的特征、模型、评估 Python 代码
├── outputs/                   # 实验指标、结果图；模型和批量预测不入 Git
└── reports/                   # 报告正文、展示材料
```

## 三人分工

- **A — 基线与共用模型代码**：检查数据对应和作曲家隔离划分，读取 CQT 并生成固定长度特征，完成基线训练流程。
- **B — 节奏特征与增强实验**：从音频提取 onset density、tempogram statistics，复用共用模型代码运行节奏与融合实验。
- **C — 评估与分析**：先实现指标、划分检查和结果汇总，再分析误排序曲目；核心任务完成且时间允许时负责 SSM 结构扩展。
- 报告各写自己负责的部分；结论、局限与展示共同完成。数据准备不单独占一个人的全部工作。

## 环境

建议先使用 Python 3.11 或 3.12 创建独立虚拟环境；不要把依赖安装到系统 Python。当前机器默认 `python3` 为 3.14.7，初始化时没有修改它，也没有安装任何依赖。

在本目录执行以下命令（以已安装 Python 3.12 为例；若只有 3.11，替换首行解释器）：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

依赖清单是开发起点，**不是已经测试过的版本锁文件**。首次成功跑通后再记录实际版本。Ordinal Logistic Regression 的具体实现尚未选定，不预先绑定实现库。

## 实验约定

1. 所有方法使用相同可用曲目集合和同一份冻结划分；先确定曲目与作曲家分组，再生成片段。官方划分需要核查，不能默认满足 composer-disjoint。
2. CQT 基线使用固定长度统计特征，例如逐频率 bin 的时间均值和标准差。若混用官方与自算 CQT，先核对频率范围、帧率、幅值变换及归一化方式。
3. 节奏特征从原始音频提取；预计算 CQT 不视为原始音频的等价替代。无可用音频的曲目不能直接进入节奏对比实验。
4. 首先完成 `CQT-only`、`Rhythm-only`、`CQT + Rhythm`；SSM 是选做，不影响核心项目交付。
5. 标准化和其他可学习预处理只在训练集拟合；超参数用验证集选择，测试集不参与调参。模型选择优先看验证集 Tau-c。
6. 主指标为 Kendall's Tau-c，另报 Spearman、MAE、MSE、Acc±1。等级编码按实际元数据核验，不能仅凭猜测重映射。
7. 连续排序得分与等级尺度预测分开：Tau-c/Spearman 使用连续得分；MAE/MSE 使用等级尺度的预测。有序模型可用类别概率的期望等级；不能把未校准的潜在得分直接与 0–10 标签计算误差。Acc±1 统一约定为舍入并截断到有效等级后的预测与真值相差不超过 1；实现时记录舍入规则。
8. Ridge 是把等级编码作为数值的对照，平方误差隐含等距编码假设；不要将其描述为完全不假设等级间距。MAE/MSE 也按该编码解释，排序结论以 Tau-c 为主。

## 协作接口

- 用数据集稳定的 recording 标识关联音频、CQT、特征和预测，并另外保留 piece 与 composer 分组标识；实际字段映射待读取元数据后确定。
- 每个特征产物保存曲目标识顺序和特征名称，不依赖目录遍历顺序拼接。
- A、B 共用同一个训练入口；C 共用同一个评估入口，避免各自算出不同口径。
- 实验结果记录特征组、模型与超参数、划分版本、随机种子、样本数和指标。不得填写虚构或示例成绩冒充真实结果。
- 每个人使用自己的功能分支，例如 `baseline`、`rhythm`、`evaluation`；合并前检查差异与测试。此次初始化没有 commit、push 或修改远端配置。

## 下一步

先核查小批数据与标签，再实现 CQT 基线、节奏特征、指标测试；不要先搭建复杂框架。具体代码位置见 [src/README.md](src/README.md)。

## 参考入口

- [MIREX task](https://music-ir.org/mirex/wiki/2026:Music_Performance_Difficulty_Prediction)
- [PSyllabus dataset](https://zenodo.org/records/14794592)
- [Author code](https://github.com/pramoneda/audio-difficulty)

源数据及录音遵守对应来源的使用与分发条件，Git 仓库不重新分发原始数据。
