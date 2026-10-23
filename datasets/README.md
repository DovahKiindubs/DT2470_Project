# 数据目录说明

项目使用 PSyllabus 数据集进行钢琴演奏难度预测。metadata 共包含 7,901 条记录，难度标签字段为 `ps`，取值为 0–10。

## 最终实验数据

最终十组实验使用冻结共同子集 `cqt_rhythm_common_v1`：

- 训练集：4,089 首
- 验证集：2,040 首
- 测试集：1,557 首
- 总计：7,686 首

划分文件位于：

```text
splits/cqt_rhythm_common_v1.json
splits/cqt_rhythm_common_v1_manifest.json
```

该子集继承原严格划分，只过滤缺少配对节奏特征或待核验的录音，不重新分配 train/val/test。

## 目录结构

```text
metadata/
  new_clean_data.json        曲目metadata与难度标签
  split_audio.json           数据集原始划分信息
splits/
  cqt_rhythm_common_v1.json  最终冻结划分
  *_manifest.json            逐曲文件哈希与成员记录
features/
  cqt/                       逐曲CQT序列，88×T，约5 fps
  rhythm_windows/            逐曲9通道节奏窗口序列
  cqt_hard_windows.npz       528维CQT汇总特征
  rhythm_stats.npz           410维节奏汇总特征
  structure_stats.npz        73维结构相关统计特征
  *_normalization.npz        仅由训练集计算的归一化参数
audio/                       可选原始音频，本项目包默认不包含
```

## 获取与放置

PSyllabus 数据与预计算 CQT 可从项目对应的 Zenodo 数据页获取：

- https://zenodo.org/records/14794592
- https://music-ir.org/mirex/wiki/2026:Music_Performance_Difficulty_Prediction

下载后请保持原始文件名，将 CQT `.bin` 放入 `features/cqt/`。节奏窗口序列由音频提取，放入 `features/rhythm_windows/`，并保留同目录的 `index.npz`。

## 注意事项

- `cqt/`、`rhythm_windows/` 和原始音频体积较大，默认由 `.gitignore` 排除。
- metadata、split、manifest 和特征缓存带有哈希校验；不要静默替换内容。
- 当前节奏与CQT对齐基于名义时间轴和共同起点，并未逐曲进行波形级验证。
- 重新提取特征时应创建新版本，不覆盖冻结实验使用的文件。
