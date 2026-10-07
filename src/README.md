# 代码放置约定

保持少量、直接的 Python 文件，不搭建额外框架。**当前尚无算法实现**；下面是建议文件名，不是已经可运行的命令。

- `features_cqt.py`：CQT 的逐频率统计特征。
- `features_rhythm.py`：从音频提取 onset density、tempogram statistics。
- `train.py`：共用的训练与预测流程，支持实际选定的回归模型和特征组。
- `evaluate.py`：统一计算 Tau-c、Spearman、MAE、MSE、Acc±1。
- `features_structure.py`：可选的 chroma/SSM 特征，仅在核心任务完成后添加。

A 主要负责 CQT 与训练代码，B 负责节奏特征并复用训练入口，C 负责评估与测试，并按时间决定是否实现结构特征。

各脚本以 `Project/` 为运行根目录，用相对路径或显式参数，不硬编码个人电脑绝对路径。探索先放 `notebooks/`，稳定逻辑再放入这里。

测试放到 `tests/`。优先检查空输入、曲目顺序、特征长度、缺失值、划分泄漏和指标边界情况；常数预测导致相关系数不可定义时应明确记录，不伪装成有效零分。
