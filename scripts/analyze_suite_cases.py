"""对符合 rerun_v1 布局的 suite 做 validation 等级统计、错误案例与 SSM 实例报告。

期望布局（与 configs/*.json 的 experiments 一致）：
  outputs/<suite>/<group>/<folder>/seed_<seed>_<run>/artifacts/predictions_val.json

用法：
  python scripts/analyze_suite_cases.py --suite rerun_v1
  python scripts/analyze_suite_cases.py --config configs/rerun_v1.json --seed 42 --run-name run_01
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from data.audio_io import load_audio  # noqa: E402
from features.cqt import ArrayUnpickler  # noqa: E402
from features.structure import MAX_FRAMES, NOVELTY_HALF, _foote_novelty, _hop_length, _time_pool  # noqa: E402


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text(encoding='utf-8'))
    if 'suite' not in cfg or 'experiments' not in cfg:
        raise ValueError(f'Config must have suite + experiments: {path}')
    return cfg


def discover_runs(cfg: dict, seed: int, run_name: str):
    """返回 {exp_id: predictions_val.json Path} 与 hgb_ids 列表（按 config 顺序）。"""
    suite = cfg['suite']
    suite_dir = ROOT / 'outputs' / suite
    runs, hgb_ids, missing = {}, [], []
    for exp in cfg['experiments']:
        eid, group, folder = exp['id'], exp['group'], exp['folder']
        path = suite_dir / group / folder / f'seed_{seed}_{run_name}' / 'artifacts' / 'predictions_val.json'
        if path.exists():
            runs[eid] = path
            if group == 'hgb':
                hgb_ids.append(eid)
        else:
            missing.append(str(path.relative_to(ROOT)))
    if not runs:
        raise FileNotFoundError(f'No predictions under outputs/{suite} (seed={seed}, run={run_name})')
    if missing:
        print(f'Warning: {len(missing)} experiment prediction files missing (skipped)')
    return runs, hgb_ids


def load_preds(runs: dict):
    tables, y_ref = {}, None
    for name, path in runs.items():
        rows = json.loads(path.read_text(encoding='utf-8'))
        by_id = {r['id']: float(r['score']) for r in rows}
        y = {r['id']: int(r['true_grade']) for r in rows}
        if y_ref is None:
            y_ref = y
        elif y != y_ref:
            raise ValueError(f'label mismatch: {name}')
        tables[name] = by_id
    ids = sorted(y_ref)
    Y = np.asarray([y_ref[i] for i in ids], dtype=np.float64)
    P = {n: np.asarray([tables[n][i] for i in ids], dtype=np.float64) for n in runs}
    return ids, Y, P


def band(y):
    if y <= 3:
        return 'Low0-3'
    if y <= 6:
        return 'Mid4-6'
    return 'High7-10'


def grade_stats(Y, P):
    stats = {}
    for name, pred in P.items():
        err = pred - Y
        rows = []
        for g in range(11):
            m = Y == g
            if not m.any():
                continue
            e = err[m]
            rows.append({
                'grade': g, 'n': int(m.sum()),
                'mae': float(np.mean(np.abs(e))), 'bias': float(np.mean(e)),
                'acc_pm1': float(np.mean(np.abs(e) <= 1.0)),
            })
        bands = []
        for bname, lo, hi in (('Low0-3', 0, 3), ('Mid4-6', 4, 6), ('High7-10', 7, 10)):
            m = (Y >= lo) & (Y <= hi)
            e = err[m]
            bands.append({
                'band': bname, 'n': int(m.sum()),
                'mae': float(np.mean(np.abs(e))), 'bias': float(np.mean(e)),
                'acc_pm1': float(np.mean(np.abs(e) <= 1.0)),
            })
        stats[name] = {
            'by_grade': rows, 'by_band': bands,
            'overall_mae': float(np.mean(np.abs(err))),
            'overall_bias': float(np.mean(err)),
        }
    return stats


def plot_grade(stats, models, path_mae, path_bias, suite):
    if not models:
        return
    grades = [r['grade'] for r in stats[models[0]]['by_grade']]
    xticks = list(range(11))
    for path, key, ylab, title, hline in (
        (path_mae, 'mae', 'MAE', f'{suite} HGB MAE by grade', False),
        (path_bias, 'bias', 'mean(pred-y)', f'{suite} HGB bias by grade', True),
    ):
        fig, ax = plt.subplots(figsize=(9, 4))
        for name in models:
            ax.plot(grades, [r[key] for r in stats[name]['by_grade']], marker='o', label=name)
        if hline:
            ax.axhline(0, color='k', lw=0.8)
        ax.set_xticks(xticks)
        ax.set_xlabel('true grade'); ax.set_ylabel(ylab); ax.legend(fontsize=8)
        ax.set_title(title); ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def timing(key, meta):
    m = meta.get(key, {})
    cqt_sec = mp3_sec = None
    cqt_path = ROOT / 'datasets/features/cqt' / f'{key}.bin'
    if cqt_path.exists():
        with cqt_path.open('rb') as f:
            cqt_sec = float(ArrayUnpickler(f).load().shape[1]) / 5.0
    mp3 = ROOT / 'datasets/audio/mp3_pieces' / f'{key}.mp3'
    if mp3.exists():
        import soundfile as sf
        mp3_sec = float(sf.info(str(mp3)).duration)
    bounds = None
    if 'start' in m and 'end' in m:
        bounds = ('start_end', float(m['start']), float(m['end']), float(m['end']) - float(m['start']))
    elif 'only_end' in m:
        bounds = ('only_end', 0.0, float(m['only_end']), float(m['only_end']))
    elif 'only_start' in m and isinstance(m.get('duration'), (int, float)):
        bounds = ('only_start', float(m['only_start']), float(m['duration']),
                  float(m['duration']) - float(m['only_start']))
    return {
        'meta_duration': m.get('duration'), 'bounds': bounds,
        'cqt_sec': cqt_sec, 'mp3_sec': mp3_sec,
        'align_risk': cqt_sec is not None and mp3_sec is not None and abs(cqt_sec - mp3_sec) > 5.0,
    }


def pick_cases(ids, Y, P, hgb_ids, k=3):
    if not hgb_ids:
        raise ValueError('Need at least one HGB experiment for case picking')
    # 精确优先，避免 hgb_cqt 匹配到 hgb_cqt_rhythm
    base_id = next((x for x in hgb_ids if x == 'hgb_cqt'), None)
    if base_id is None:
        base_id = next(
            (x for x in hgb_ids if 'rhythm' not in x and 'structure' not in x),
            hgb_ids[0],
        )
    rh_id = next((x for x in hgb_ids if 'rhythm' in x and 'structure' not in x), None)
    st_id = next((x for x in hgb_ids if 'structure' in x and 'rhythm' not in x), None)
    full_id = next((x for x in hgb_ids if 'rhythm' in x and 'structure' in x), None)

    base = P[base_id]
    abs_base = np.abs(base - Y)

    def top(mask, score, n=k, bigger=True):
        idx = np.where(mask)[0]
        if idx.size == 0:
            return []
        order = np.argsort(score[idx])
        if bigger:
            order = order[::-1]
        return [ids[i] for i in idx[order[:n]]]

    chosen = {'baseline_large_err': top(np.ones(len(ids), bool), abs_base, 4)}
    if st_id:
        d = np.abs(P[st_id] - Y) - abs_base
        chosen['structure_improved'] = top(d < -0.5, -d, 3)
        chosen['structure_worsened'] = top(d > 0.5, d, 3)
    if rh_id:
        d = np.abs(P[rh_id] - Y) - abs_base
        chosen['rhythm_improved'] = top(d < -0.5, -d, 3)
        chosen['rhythm_worsened'] = top(d > 0.5, d, 3)
    if full_id:
        chosen['full_still_bad'] = top(np.abs(P[full_id] - Y) >= 2.0, np.abs(P[full_id] - Y), 3)

    seen, ordered = set(), []
    for cat, keys in chosen.items():
        for key in keys:
            if key not in seen:
                seen.add(key)
                ordered.append((cat, key))
    roles = {'base': base_id, 'rhythm': rh_id, 'structure': st_id, 'full': full_id}
    return chosen, ordered, roles


def md_table(headers, rows):
    lines = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    for r in rows:
        lines.append('| ' + ' | '.join(r) + ' |')
    return '\n'.join(lines)


def write_error_report(suite, out_err, ids, Y, P, stats, meta, chosen, ordered, hgb_ids, roles, all_ids):
    out_err.mkdir(parents=True, exist_ok=True)
    plot_grade(stats, hgb_ids, out_err / 'grade_mae.png', out_err / 'grade_bias.png', suite)
    id_to_i = {k: i for i, k in enumerate(ids)}

    lines = [
        f'# 逐曲错误案例分析（{suite} validation）',
        '',
        '仅 validation；未评估 test。试听栏默认为 pending，不能视为机制已证实。',
        '',
        '## 1. 整体等级统计',
        '',
        '### 1.1 HGB 分等级 MAE / bias',
        '',
        '![MAE](grade_mae.png)',
        '',
        '![bias](grade_bias.png)',
        '',
    ]
    band_rows = []
    for name in hgb_ids:
        for b in stats[name]['by_band']:
            band_rows.append([
                name, b['band'], str(b['n']), f"{b['mae']:.3f}", f"{b['bias']:+.3f}", f"{b['acc_pm1']:.3f}",
            ])
    lines += ['### 1.2 等级带汇总（HGB）', '', md_table(['model', 'band', 'n', 'MAE', 'bias', 'acc±1'], band_rows), '']

    base_id, full_id = roles['base'], roles['full']
    base_bands = {b['band']: b for b in stats[base_id]['by_band']}
    lines += [
        '### 1.3 简要结论',
        '',
        f"- 基线 `{base_id}` 整体 MAE={stats[base_id]['overall_mae']:.3f}，"
        f"bias={stats[base_id]['overall_bias']:+.3f}。",
        f"- 分带 MAE：Low {base_bands['Low0-3']['mae']:.3f} / "
        f"Mid {base_bands['Mid4-6']['mae']:.3f} / High {base_bands['High7-10']['mae']:.3f}。",
        f"- 分带 bias：Low {base_bands['Low0-3']['bias']:+.3f} / "
        f"Mid {base_bands['Mid4-6']['bias']:+.3f} / High {base_bands['High7-10']['bias']:+.3f} "
        f"（正=高估，负=低估）。",
    ]
    if full_id:
        fb = {b['band']: b for b in stats[full_id]['by_band']}
        lines.append(
            f"- 全加 `{full_id}` High MAE {fb['High7-10']['mae']:.3f}，bias {fb['High7-10']['bias']:+.3f}。"
        )
    lines += [
        '- 若 Low 正 bias、High 负 bias，则为向中间收缩的等级偏好。',
        '',
        '## 2. 筛选案例',
        '',
        f'筛选规则见 `../PLAN.md`；本报告共 {len(ordered)} 首（类别间去重后）。',
        '',
    ]

    for cat, key in ordered:
        i = id_to_i[key]
        y = int(Y[i])
        t = timing(key, meta)
        pred_rows = [[n, f'{float(P[n][i]):.3f}', f'{float(P[n][i]) - y:+.3f}'] for n in all_ids]
        bound = '无'
        if t['bounds']:
            kind, a, b, L = t['bounds']
            bound = f'{kind} [{a:.1f},{b:.1f}] len={L:.1f}s'
        lines += [
            f'### {key}', '',
            f'- 类别：`{cat}`；真值 **{y}**（{band(y)}）',
            f"- 时长：CQT={t['cqt_sec'] and round(t['cqt_sec'],1)}s；"
            f"meta_duration={t['meta_duration']}；MP3={t['mp3_sec'] and round(t['mp3_sec'],1)}s；"
            f"边界={bound}；对齐风险={'是' if t['align_risk'] else '否'}",
            '', md_table(['experiment', 'pred', 'err'], pred_rows), '',
            '- 人工试听：`pending`', '- 假设原因：`pending`', '',
        ]

    lines += [
        '## 3. 小结', '',
        '- 上表案例供精听；未填试听前不得写机制结论。',
        '- 对齐风险=是：CQT 与 MP3 时长差>5s，误差可能含数据问题。',
        '',
        '### 各类入选（含与他类重复前）', '',
    ]
    for cat, keys in chosen.items():
        lines.append(f'- `{cat}` ({len(keys)})：' + '; '.join(keys))
    lines.append('')
    (out_err / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')


def chroma_ssm(y, sr):
    import librosa
    hop = _hop_length(sr)
    chroma = np.asarray(librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop), dtype=np.float64)
    chroma = chroma / np.maximum(np.linalg.norm(chroma, axis=0, keepdims=True), 1e-8)
    chroma = _time_pool(chroma, MAX_FRAMES)
    ssm = chroma.T @ chroma
    novelty = _foote_novelty(ssm, half=NOVELTY_HALF)
    dur = float(y.size) / float(sr)
    times = np.linspace(0.0, dur, ssm.shape[0], endpoint=False)
    return ssm, novelty, times, dur


def best_repeat_pair(ssm, times, dur, win_sec=8.0):
    n = ssm.shape[0]
    best = (-1.0, 0, 0)
    for i in range(n):
        for j in range(i + max(3, n // 10), n):
            v = float(ssm[i, j])
            if v > best[0]:
                best = (v, i, j)
    _, i, j = best
    half = max(1, int(round(win_sec / max(dur / n, 1e-6) / 2)))

    def clip(c):
        a = max(0, c - half)
        b = min(n - 1, c + half)
        return float(times[a]), float(times[min(b, n - 1)] + dur / n)

    return best[0], clip(i), clip(j)


def pick_ssm_default(ids, Y, P, meta, ordered_err, roles, n=6):
    """原规则：structure 改善/变差各2 + chapters + 误差列表补齐。"""
    base_id, st_id = roles['base'], roles['structure']
    picks = []
    if st_id:
        d_st = np.abs(P[st_id] - Y) - np.abs(P[base_id] - Y)
        for k in [ids[i] for i in np.argsort(d_st)[:2]]:
            picks.append((k, 'structure_improved'))
        for k in [ids[i] for i in np.argsort(-d_st)[:2]]:
            if k not in {p[0] for p in picks}:
                picks.append((k, 'structure_worsened'))
    for key in ids:
        ch = meta.get(key, {}).get('YT_chapters')
        if isinstance(ch, list) and len(ch) >= 4:
            if key not in {p[0] for p in picks}:
                picks.append((key, 'has_chapters'))
            break
    for cat, key in ordered_err:
        if len(picks) >= n:
            break
        if key not in {p[0] for p in picks}:
            picks.append((key, f'from_error:{cat}'))
    return picks[:n]


def pick_ssm_from_errors(ordered_err, n=6):
    """直接从错误案例去重列表取前 n 首（预测偏差筛选集）。"""
    picks = []
    for cat, key in ordered_err:
        if key not in {p[0] for p in picks}:
            picks.append((key, f'from_error:{cat}'))
        if len(picks) >= n:
            break
    return picks


def pick_ssm_merged(ids, Y, P, meta, ordered_err, roles, error_n=6):
    """structure/chapters 对照 + 误差列表补齐（去重，统一写入 ssm_cases）。"""
    picks = pick_ssm_default(ids, Y, P, meta, ordered_err, roles, n=6)
    seen = {p[0] for p in picks}
    for cat, key in ordered_err:
        if key in seen:
            continue
        picks.append((key, f'from_error:{cat}'))
        seen.add(key)
        # 至少纳入 error 列表前 error_n 首中尚未覆盖者
        covered_err = sum(1 for k, _ in pick_ssm_from_errors(ordered_err, n=error_n) if k in seen)
        if covered_err >= error_n and len(picks) >= 6:
            break
    # 确保 error 前 error_n 首都在
    for key, reason in pick_ssm_from_errors(ordered_err, n=error_n):
        if key not in seen:
            picks.append((key, reason))
            seen.add(key)
    return picks


def write_ssm_cases(suite, out_ssm, ids, Y, P, meta, ordered_err, hgb_ids, roles,
                    picks=None, title_note=''):
    out_ssm.mkdir(parents=True, exist_ok=True)
    id_to_i = {k: i for i, k in enumerate(ids)}
    if picks is None:
        picks = pick_ssm_from_errors(ordered_err, n=10)

    report = [
        f'# SSM 音乐实例核验（{suite} validation）', '',
        '自动相似候选不代表已确认乐句；novelty 为现有启发式，**非完整 Foote**。',
        '人工试听未完成前结论只能是 pending。时间轴相对当前 MP3。', '',
    ]
    if title_note:
        report[3:3] = [title_note, '']

    for ci, (key, reason) in enumerate(picks, 1):
        i = id_to_i[key]
        y = int(Y[i])
        case_dir = out_ssm / f'case_{ci:02d}'
        if case_dir.exists():
            import shutil
            shutil.rmtree(case_dir)
        case_dir.mkdir(parents=True, exist_ok=True)
        mp3 = ROOT / 'datasets/audio/mp3_pieces' / f'{key}.mp3'
        tinfo = timing(key, meta)
        pred_line = ', '.join(f'{n}={P[n][i]:.2f}' for n in hgb_ids)
        status, note = 'ok', ''
        if not mp3.exists():
            status, note = 'missing_audio', '无本地 MP3，跳过图/试听片段'
        else:
            try:
                y_wav, sr = load_audio(mp3, normalize=True)
                ssm, novelty, times, dur = chroma_ssm(y_wav, sr)
                sim, (a0, a1), (b0, b1) = best_repeat_pair(ssm, times, dur)
                fig, ax = plt.subplots(figsize=(5.5, 5))
                ax.imshow(ssm, origin='lower', aspect='auto', extent=[0, dur, 0, dur], cmap='magma')
                peaks = np.where(
                    (novelty[1:-1] > novelty[:-2]) & (novelty[1:-1] > novelty[2:])
                    & (novelty[1:-1] > novelty.mean() + novelty.std())
                )[0] + 1
                for p in peaks[:8]:
                    ax.axhline(times[p], color='#4FC3F7', ls='--', lw=1.2, alpha=0.9)
                    ax.axvline(times[p], color='#4FC3F7', ls='--', lw=1.2, alpha=0.9)
                ax.set_title(f'{key[:48]}\nSSM pooled→{ssm.shape[0]}')
                ax.set_xlabel('s'); ax.set_ylabel('s')
                # Unique filename so Markdown preview cannot reuse a stale case_XX/ssm.png cache.
                import re as _re
                slug = _re.sub(r'[^A-Za-z0-9]+', '_', key).strip('_')[:60]
                png_name = f'ssm_{slug}.png'
                fig.tight_layout(); fig.savefig(case_dir / png_name, dpi=120); plt.close(fig)
                import soundfile as sf
                for tag, (t0, t1) in (('repeat_1', (a0, a1)), ('repeat_2', (b0, b1))):
                    sf.write(str(case_dir / f'{tag}.wav'), y_wav[int(t0 * sr):int(t1 * sr)], sr)
                note = (
                    f'自动相似段 [{a0:.1f}-{a1:.1f}s] vs [{b0:.1f}-{b1:.1f}s]，'
                    f'sim={sim:.3f}；novelty峰约 {len(peaks)} 个'
                )
                json.dump({
                    'id': key, 'reason': reason, 'y': y, 'ssm_png': png_name,
                    'preds_hgb': {n: float(P[n][i]) for n in hgb_ids},
                    'timing': tinfo, 'repeat': {'sim': sim, 'a': [a0, a1], 'b': [b0, b1]},
                    'listening': 'pending', 'verdict': 'pending',
                }, (case_dir / 'case.json').open('w', encoding='utf-8'), ensure_ascii=False, indent=2)
            except Exception as exc:
                status, note = 'failed', str(exc)
                png_name = None

        report += [
            f'## {ci}. {key}', '',
            f'真值 {y}；HGB：{pred_line}',
            f"时长：CQT={tinfo['cqt_sec'] and round(tinfo['cqt_sec'],1)}s，"
            f"MP3={tinfo['mp3_sec'] and round(tinfo['mp3_sec'],1)}s，"
            f"meta_duration={tinfo['meta_duration']}；对齐风险={'是' if tinfo['align_risk'] else '否'}",
            f'状态：`{status}`。{note}', '',
        ]
        png_path = None
        if status == 'ok':
            pngs = sorted(case_dir.glob('ssm_*.png'))
            if pngs:
                png_path = pngs[0].name
        if png_path:
            report += [
                f'![SSM](case_{ci:02d}/{png_path})', '',
                f'候选重复：[repeat_1](case_{ci:02d}/repeat_1.wav) / '
                f'[repeat_2](case_{ci:02d}/repeat_2.wav)', '',
            ]
        report += ['人工核验：`listening=pending`，`verdict=pending`', '']

    report += [
        '## 表述限制', '',
        '- 未试听完成前，不得写“结构特征已验证”。',
        '- novelty 非标准 Foote checkerboard。',
        '- 若对齐风险=是，SSM 时间点不能直接当成作品结构位置。', '',
    ]
    (out_ssm / 'RESULTS.md').write_text('\n'.join(report), encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', default=None, help='suite 名，默认读对应 configs/<suite>.json')
    p.add_argument('--config', type=Path, default=None, help='实验配置 JSON（含 suite/experiments）')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--run-name', default='run_01')
    p.add_argument('--metadata', type=Path, default=ROOT / 'datasets/metadata/new_clean_data.json')
    p.add_argument('--skip-ssm', action='store_true')
    p.add_argument('--skip-error-report', action='store_true',
                   help='只生成 SSM，不覆盖 error_cases/RESULTS.md')
    p.add_argument('--ssm-error-n', type=int, default=10,
                   help='ssm_cases 选取 error_cases 去重列表前 n 首')
    args = p.parse_args()

    if args.config is None:
        suite = args.suite or 'rerun_v1'
        cfg_path = ROOT / 'configs' / f'{suite}.json'
    else:
        cfg_path = args.config
    cfg = load_config(cfg_path)
    suite = args.suite or cfg['suite']
    if suite != cfg['suite']:
        raise ValueError(f'--suite {suite} != config suite {cfg["suite"]}')

    runs, hgb_ids = discover_runs(cfg, args.seed, args.run_name)
    ids, Y, P = load_preds(runs)
    meta = json.loads(args.metadata.read_text(encoding='utf-8'))
    stats = grade_stats(Y, P)
    chosen, ordered, roles = pick_cases(ids, Y, P, hgb_ids)

    out = ROOT / 'outputs' / suite / 'analysis'
    # copy/update PLAN pointer once
    plan = out / 'PLAN.md'
    if not plan.exists():
        plan.parent.mkdir(parents=True, exist_ok=True)
        plan.write_text(
            f'# {suite} 案例分析\n\n'
            f'由 `scripts/analyze_suite_cases.py --suite {suite}` 生成。\n'
            '通用约定见仓库内任一 suite 的 analysis/PLAN 模板或本脚本 docstring。\n',
            encoding='utf-8',
        )

    if not args.skip_error_report:
        write_error_report(
            suite, out / 'error_cases', ids, Y, P, stats, meta, chosen, ordered,
            hgb_ids, roles, list(runs),
        )
    written = []
    if not args.skip_ssm:
        picks = pick_ssm_from_errors(ordered, n=args.ssm_error_n)
        write_ssm_cases(
            suite, out / 'ssm_cases', ids, Y, P, meta, ordered, hgb_ids, roles,
            picks=picks,
            title_note=(
                f'选自 validation **错误预测**案例（error_cases 去重列表前 {len(picks)} 首）；'
                '不标注内部筛选标签。'
            ),
        )
        written.append(f'outputs/{suite}/analysis/ssm_cases/RESULTS.md')
    parts = []
    if not args.skip_error_report:
        parts.append(f'outputs/{suite}/analysis/error_cases/RESULTS.md')
    parts.extend(written)
    print('Done: ' + (' | '.join(parts) if parts else '(nothing)'))


if __name__ == '__main__':
    main()
