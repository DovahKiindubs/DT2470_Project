# PSyllabus 数据目录

## 已整理的文件

- `metadata/new_clean_data.json`：原始曲目元数据与标签，7,901 条记录。
- `metadata/split_audio.json`：作者提供的 5 折曲目级划分，保持原样，当前主实验未使用。
- `features/cqt/`：7,901 个 CQT `.bin` 文件，文件名未修改。
- `features/cqt_stats.npz`：从每首曲子的 CQT 提取的 176 维统计特征，即 88 个频率 bin 的时间均值与标准差。
- `audio/mp3_pieces/`：本地原始 MP3（约 7703 首，stem 与元数据 key 精确匹配）；不落盘全量 WAV。
- `audio/reports/audio_qc.json`：音频对齐与质检报告（见下节）。
- `splits/cqt_composer_split.json`：当前 Ridge 与 HGB 实验使用的严格划分（全量 CQT）。
- `splits/audio_composer_subset.json`：QC 通过音频 ∩ 上述严格划分，供节奏特征 ablation。
- `inventory.json`：下载文件校验值、CQT 完整性和初始数据检查记录。

## 原始音频与共同子集

节奏特征（onset density、tempogram 等）需要原始音频。加载协议由 `src/audio_io.py` 固定：

- 保留文件原生采样率、单声道、float32；
- 峰值归一化到 `0.99`（全静音不缩放）；
- 不截断、不 trim silence、不做 R128 响度归一化；
- 统一走 `load_audio()`；节奏特征应用**秒**为单位的 hop/窗，勿写死 sample hop。

生成报告：

```bash
python src/preprocess_audio.py
```

产出：

- `audio/reports/audio_qc.json`：对齐摘要、QC 统计、`failed`、`ok`
- `splits/audio_composer_subset.json`：通过曲目 ∩ `cqt_composer_split`（含相对全量 split 的 missing）

硬失败（剔除）：无法解码、时长 `< 3 s`、全静音、含 NaN/Inf。软警告（保留）：与 meta duration 相对误差 `> 10%`、缺 meta duration、极低 RMS、削波、时长 `> 600 s`。

**实验纪律：** 报告 CQT vs +rhythm 时，必须在 `splits/audio_composer_subset.json` 上重训/重评 CQT-only；禁止把全量 7901 的 CQT 分数与音频子集上的节奏分数直接比较。划分边界仍沿用 `cqt_composer_split.json` 的分组，不重新划分。

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
