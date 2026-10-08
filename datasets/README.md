# PSyllabus 数据目录

## 已整理的文件

- `metadata/new_clean_data.json`：原始曲目元数据与标签，7,901 条记录。
- `metadata/split_audio.json`：作者提供的 5 折曲目级划分，保持原样，当前主实验未使用。
- `features/cqt/`：7,901 个 CQT `.bin` 文件，文件名未修改。
- `features/cqt_stats.npz`：从每首曲子的 CQT 提取的 176 维统计特征，即 88 个频率 bin 的时间均值与标准差。
- `audio/`：目前没有完整原始音频；元数据中的链接不等于音频文件。
- `splits/cqt_composer_split.json`：当前 Ridge 与 HGB 实验使用的严格划分。
- `inventory.json`：下载文件校验值、CQT 完整性和初始数据检查记录。

## 读取提示

元数据是以曲目名称为键的 JSON 对象。CQT 文件名去掉 `.bin` 后与元数据键完全匹配，可据此关联；不要根据目录遍历顺序拼接标签。

主难度字段 `ps` 的实际值为字符串 `0` 至 `10`，读取时显式转成整数。`composer` 可用于初步作曲家分组。文件键匹配不等于已经检查所有同曲不同录音、乐章与作品之间的关系，训练前仍需复核。

**CQT 文件是 `.bin`，不是 `.npy`。** 抽查头信息显示其包含 pickle 序列化的 NumPy 数组，不能直接假设为 NumPy 的 `.npy` 格式。本次只核验文件头与完整性，没有执行 pickle 反序列化或检查全部数值内容。后续仅对已确认可信来源的文件使用对应加载方法；pickle 可能执行代码，不能加载不可信文件。

节奏特征仍需要原始音频。MIDI/piano roll 仅作为下载素材保留，未加入 proposal 的 audio-only 主实验，也未生成替代音频。

## 当前主实验：严格划分

当前 Ridge 和 Histogram Gradient Boosting 的验证结果使用 `splits/cqt_composer_split.json`，没有使用官方 `split_audio.json`。

该划分采用固定 seed=42，先以分组方式留出约 20% 测试组，再从剩余组中留出约 25% 验证组。实际数量为：

- train：4,211 条；
- validation：2,095 条；
- test：1,595 条。

分组规则同时约束：

1. 规范化后的同一 `composer` 字符串不能跨 train/validation/test；
2. 相同 YouTube 视频 ID 不能跨集合，避免多个条目引用同一录音而造成直接泄漏。

因此，当前划分是 **composer-disjoint + recording-link-disjoint**。它比官方划分更严格，主要用于评估模型对未见作曲家与未见录音的泛化。目前 Ridge 和 HGB 只查看了 validation 表现，test 尚未评估。

局限是 `composer` 仅做 Unicode、大小写和空白规范化，没有进行音乐学意义上的作曲家别名消歧；相同作品的不同标题或不同录音也没有完整的作品级归并。因此这里的“严格”只指上述已执行规则，不代表完全解决所有作品身份问题。

## 官方 5 折划分：较宽松的辅助协议

`metadata/split_audio.json` 包含 5 折，每折 train / validation / test 分别为 4,736 / 1,576 / 1,589 条。各集合之间没有相同曲目键，但不是作曲家隔离划分。

按元数据中的 `composer` 字符串和 YouTube 视频 ID 检查：

- train/test 每折有 357–381 个重叠作曲家字符串；
- train/test 每折有 46–62 个重叠 YouTube 视频 ID；
- train/validation 和 validation/test 也存在同类重叠。

因此，官方划分适合做与作者流程接近的**辅助 5-fold 实验**，可以报告 5 折均值和标准差；但它评估的是较宽松的曲目级预测，不能声称 composer-disjoint。共享 YouTube ID 还可能带来直接录音泄漏，所以若使用官方划分，报告中必须明确说明这一点。

推荐报告方式：

- **Primary**：当前严格划分，回答跨作曲家、跨录音泛化问题；
- **Secondary**：官方 5 折，作为较宽松的可比性实验。

两套协议的结果不能直接混合，也不能因为官方划分分数可能更高就替代严格结果。


来源：[PSyllabus 发布页](https://zenodo.org/records/14794592)。按来源授权使用，不在本仓库重新分发录音或原始数据。
