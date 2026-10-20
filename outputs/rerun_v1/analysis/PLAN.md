# Suite 案例分析计划（通用）

适用于任何与 `rerun_v1` 同构的 suite：

- 配置：`configs/<suite>.json`（含 `suite`、`experiments[].{id,group,folder}`）
- 预测：`outputs/<suite>/<group>/<folder>/seed_<seed>_<run>/artifacts/predictions_val.json`

仅 **validation**；试听结论需人工填写。不碰 test、不重训。

## 流程

1. **整体等级统计**（分等级 MAE / bias / Acc±1）→ `error_cases/`
2. **逐曲错误案例** → `error_cases/RESULTS.md`
3. **SSM 实例** → `ssm_cases/RESULTS.md`（错误预测曲目，不写内部筛选标签）

## 逐曲错误案例：筛选规则

以 HGB 为主（有则用）：

| 代号 | 含义 | 规则（validation） |
|------|------|-------------------|
| `baseline_large_err` | 基线大误差 | `hgb_cqt` 的 \|pred−y\| 最大的 **4** 首 |
| `structure_improved` | 加结构后变好 | \|err_structure\|−\|err_cqt\| **< −0.5**，改善最大的 **3** 首 |
| `structure_worsened` | 加结构后变差 | 差值 **> +0.5**，变差最大的 **3** 首 |
| `rhythm_improved` | 加节奏后变好 | 相对 `hgb_cqt`，节奏专用实验同理，**3** 首 |
| `rhythm_worsened` | 加节奏后变差 | 同上，**3** 首 |
| `full_still_bad` | 全加仍差 | `hgb_cqt_rhythm_structure` 的 \|err\| **≥ 2**，最大的 **3** 首 |

说明：

- `err = pred − y`；比较用绝对误差差：`Δ = \|err_新\| − \|err_基线\|`（负=变好）。
- 各类入选后 **按上表顺序去重**：一首只保留第一次出现的类别。
- 某类实验缺失（如没有 structure）则跳过该类。
- 报告里每首附：九组预测、CQT/MP3/边界时长、对齐风险（CQT 与 MP3 差 >5s）。

## SSM 实例：`ssm_cases/`

从 `error_cases` 去重列表取前 **N** 首（默认 **10**）错误预测曲目，生成 SSM 图与试听片段。  
**报告正文不写**内部筛选代号，只呈现为错误预测案例。

**注意：** 脚本重跑会覆盖 `ssm_cases/RESULTS.md` 与 case 文件；已填听辨请先备份。

## 命令

```bash
python scripts/analyze_suite_cases.py --suite rerun_v1
python scripts/analyze_suite_cases.py --suite rerun_v1 --ssm-error-n 10 --skip-error-report
```
