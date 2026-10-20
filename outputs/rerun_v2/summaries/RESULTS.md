# 重跑结果与进度

来源：每次运行的run.json和artifacts/metrics.json。不扫描历史归档，不按高分隐藏失败。
running在进程被强杀后可能残留；确认日志/进程后使用新run名重试。

## hgb

- **hgb_cqt** — completed · `hgb/cqt/seed_42_run_01` · kendall_tau_c=0.6573, mae=1.3772, mse=3.1240, acc_pm1=0.6260
- **hgb_cqt_rhythm** — completed · `hgb/cqt_rhythm/seed_42_run_01` · kendall_tau_c=0.6801, mae=1.3079, mse=2.8363, acc_pm1=0.6632
- **hgb_cqt_structure** — completed · `hgb/cqt_structure/seed_42_run_01` · kendall_tau_c=0.6840, mae=1.2713, mse=2.7369, acc_pm1=0.6716
- **hgb_cqt_rhythm_structure** — completed · `hgb/cqt_rhythm_structure/seed_42_run_01` · kendall_tau_c=0.7058, mae=1.2143, mse=2.4813, acc_pm1=0.6873

## early

- **early_cqt** — completed · `early/cqt_only/seed_42_run_01` · kendall_tau_c=0.6957, mae=1.2540, mse=2.6240, acc_pm1=0.6775
- **early_rhythm** — completed · `early/rhythm_only/seed_42_run_01` · kendall_tau_c=0.6568, mae=1.4574, mse=3.3347, acc_pm1=0.5892
- **early_cqt_rhythm** — completed · `early/cqt_rhythm/seed_42_run_01` · kendall_tau_c=0.6891, mae=1.2617, mse=2.7709, acc_pm1=0.6765

## late

- **late_cqt** — completed · `late/cqt_only/seed_42_run_01` · kendall_tau_c=0.6451, mae=1.4133, mse=3.3092, acc_pm1=0.6328
- **late_rhythm** — completed · `late/rhythm_only/seed_42_run_01` · kendall_tau_c=0.6331, mae=1.4649, mse=3.4571, acc_pm1=0.5907
- **late_combined** — completed · `late/combined/seed_42_run_01` · kendall_tau_c=0.6841, mae=1.2544, mse=2.7423, acc_pm1=0.6863
