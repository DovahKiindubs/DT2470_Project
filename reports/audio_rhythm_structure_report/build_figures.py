"""Build report figures from current audio and frozen rerun_v1 validation outputs.

Run: .venv/Scripts/python.exe reports/audio_rhythm_structure_report/build_figures.py
Only writes this report directory; no feature extraction caches or audio exports.
"""
import csv
import json
import os
from pathlib import Path
import sys

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(variable, '1')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from data.audio_io import load_audio
from features.cqt import sha256
from features.rhythm import HOP_SEC, TEMPOGRAM_WIN, _binned_onset_density
from features.structure import MAX_FRAMES, NOVELTY_HALF, _foote_novelty, _time_pool
from reporting.metrics import evaluate

BLUE, ORANGE, GREEN, RED = '#2463a6', '#d38020', '#27846a', '#b94748'
MODELS = ('cqt', 'cqt_rhythm', 'cqt_structure', 'cqt_rhythm_structure')
LABELS = ('CQT', 'CQT + rhythm', 'CQT + structure', 'CQT + both')
RHYTHM_CASES = (
    ('Tan C.The Wind from A Childs Garden of Verses', 'Tan: The Wind'),
    ('Scriabin A.Prelude - op 11 no 17 Ab major', 'Scriabin: Prelude Op. 11 No. 17'),
)
STRUCTURE_CASES = (
    ('Rollin C.Bank of the Oise at Auvers', 'Rollin: Bank of the Oise at Auvers'),
    ('Ravel M.Prelude (1913)', 'Ravel: Prelude (1913)'),
)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titleweight': 'bold', 'pdf.fonttype': 42,
                     'savefig.facecolor': 'white'})
EVIDENCE = {'figure_audio': {}, 'figure_files': [], 'notes': [
    'Audio illustrations use current MP3; predictions use frozen rerun_v1.',
    'No model was trained; test predictions are never loaded.',
    'Structure figures preserve the original pooling and novelty heuristic.',
]}


def save(fig, name):
    for suffix in ('png', 'pdf'):
        path = OUT / 'figures' / f'{name}.{suffix}'
        fig.savefig(path, dpi=220, bbox_inches='tight')
        EVIDENCE['figure_files'].append(str(path.relative_to(OUT)))
    plt.close(fig)
    print(f'Saved {name}', flush=True)


def validation():
    tables = {}
    for model in MODELS:
        folder = ROOT / 'outputs/rerun_v1/hgb' / model / 'seed_42_run_01/artifacts'
        rows = json.loads((folder / 'predictions_val.json').read_text(encoding='utf-8'))
        tables[model] = {row['id']: row for row in rows}
        metrics = evaluate([r['true_grade'] for r in rows], [r['score'] for r in rows])
        saved = json.loads((folder / 'metrics.json').read_text(encoding='utf-8'))['validation']
        for key in ('kendall_tau_c', 'mae', 'mse', 'acc_pm1'):
            assert abs(metrics[key] - saved[key]) < 1e-12
        EVIDENCE.setdefault('rerun_v1_validation', {})[model] = metrics
    reference = {key: row['true_grade'] for key, row in tables['cqt'].items()}
    assert all({key: row['true_grade'] for key, row in table.items()} == reference
               for table in tables.values())
    return tables


def audio_record(key):
    path = ROOT / 'datasets/audio/mp3_pieces' / f'{key}.mp3'
    y, sr = load_audio(path, normalize=True)
    digest = sha256(path)
    old_hashes = {}
    grade = None
    for group in ('rhythm', 'structure'):
        with np.load(ROOT / f'datasets/features/{group}_stats.npz', allow_pickle=False) as d:
            lookup = dict(zip(d['ids'].tolist(), d['audio_sha256'].tolist()))
            old_hashes[group] = lookup.get(key)
            current_grade = int(d['y'][d['ids'].tolist().index(key)])
            assert grade is None or grade == current_grade
            grade = current_grade
    EVIDENCE['figure_audio'][key] = {
        'path': str(path.relative_to(ROOT)), 'seconds': len(y) / sr, 'sample_rate': sr,
        'difficulty_grade': grade,
        'current_audio_sha256': digest, 'old_audio_sha256': old_hashes,
        'same_as_rerun_v1_audio': {name: value == digest for name, value in old_hashes.items()},
    }
    print(f'Loaded {key} ({len(y) / sr:.1f}s)', flush=True)
    return y, sr


def duration_figure():
    source = ROOT / 'datasets/audio/reports/all_audio_cqt_duration_comparison.csv'
    rows = list(csv.DictReader(source.open(encoding='utf-8-sig', newline='')))
    from collections import Counter
    columns = ('切片规则', '比对结果', '本轮检查处理', '重下载状态', '新旧下载文件一致')
    EVIDENCE['audio_audit'] = {col: dict(Counter(r[col] for r in rows)) for col in columns}
    EVIDENCE['audio_audit']['rows'] = len(rows)
    EVIDENCE['audio_audit']['csv_sha256'] = sha256(source)
    present = [r for r in rows if r['比对结果'] != 'missing_audio']
    errors = np.asarray([float(r['绝对差值(秒)']) for r in present])
    assert sum(errors <= 1) == sum(r['比对结果'] == 'match' for r in present)
    EVIDENCE['audio_audit']['match_fraction_of_available'] = float(np.mean(errors <= 1))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), layout='constrained')
    labels = ['Matched (<= 1 s)', 'Still mismatched', 'Missing audio']
    counts = [sum(r['比对结果'] == state for r in rows)
              for state in ('match', 'mismatch', 'missing_audio')]
    bars = axes[0].barh(labels, counts, color=[GREEN, RED, '#9cabb9'])
    axes[0].bar_label(bars, padding=5, fmt='%d')
    axes[0].set_xlim(0, max(counts) * 1.18)
    axes[0].invert_yaxis()
    axes[0].set(xlabel='Number of metadata entries', title='A  Current audio / CQT audit')
    axes[1].hist(np.maximum(errors, .001), bins=np.geomspace(.001, max(errors.max() * 1.05, 2), 55),
                 color=BLUE, alpha=.85)
    axes[1].axvline(1, color=RED, linestyle='--', label='1 s threshold')
    axes[1].set(xscale='log', yscale='log', xlabel='Absolute duration difference (s)',
                ylabel='Number of available recordings', title='B  Duration differences (7,703 MP3s)')
    axes[1].legend(frameon=False)
    save(fig, '01_audio_quality')


def rhythm_figures():
    import librosa
    records = []
    for key, title in RHYTHM_CASES:
        y, sr = audio_record(key)
        hop = max(1, round(sr * HOP_SEC))
        envelope = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
        onsets = librosa.onset.onset_detect(onset_envelope=envelope, sr=sr,
                                           hop_length=hop, units='time')
        tg = librosa.feature.tempogram(onset_envelope=envelope, sr=sr,
                                      hop_length=hop, win_length=TEMPOGRAM_WIN)
        duration = len(y) / sr
        bins = _binned_onset_density(onsets, duration, 8)
        EVIDENCE['figure_audio'][key]['rhythm_display'] = {
            'onsets': len(onsets), 'onsets_per_sec': len(onsets) / duration,
            'eight_bin_density': bins.tolist(), 'tempogram_shape': list(tg.shape),
            'ioi_median_seconds': float(np.median(np.diff(onsets))),
            'density_bin_std': float(bins.std()),
            'tempogram_lag25_mean': float(tg[25].mean()),
            'tempogram_lag25_std': float(tg[25].std()),
        }
        title = f"{title} (grade {EVIDENCE['figure_audio'][key]['difficulty_grade']})"
        records.append((key, title, duration, hop / sr, envelope, onsets, bins, tg))
        del y
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.2), layout='constrained', sharey='row')
    for ax in axes.flat:
        ax.tick_params(axis='y', labelleft=True)
    density_limit = max(record[6].max() for record in records) * 1.15
    for col, (key, title, duration, step, env, onsets, bins, tg) in enumerate(records):
        time = np.arange(len(env)) * step
        scale = env.max() or 1
        axes[0, col].plot(time, env / scale, color=BLUE, linewidth=1)
        for onset in onsets[onsets <= 12]:
            axes[0, col].axvline(onset, color=ORANGE, linewidth=.7, alpha=.75)
        axes[0, col].set(xlim=(0, min(12, duration)), ylim=(0, 1.05),
                         xlabel='Excerpt time (s)', ylabel='Onset strength (display-normalized)',
                         title=f'{title}\nFirst 12 s; orange = detected onset')
        edges = np.linspace(0, duration, 9)
        axes[1, col].bar((edges[:-1] + edges[1:]) / 2, bins, width=duration / 8 * .87,
                        color=GREEN, alpha=.85)
        global_density = len(onsets) / duration
        axes[1, col].axhline(global_density, color=RED, linestyle='--',
                             label=f'Global: {global_density:.2f} onsets/s')
        axes[1, col].set(xlim=(0, duration), xlabel='Full recording time (s)',
                         ylim=(0, density_limit), ylabel='Detected onsets / s',
                         title='Eight equal-duration sections')
        axes[1, col].legend(frameon=False)
    save(fig, '02_onsets_density')

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout='constrained',
                             gridspec_kw={'height_ratios': [1.3, 1]})
    for col, (key, title, duration, step, env, onsets, bins, tg) in enumerate(records):
        limit = 101
        image = axes[0, col].imshow(tg[1:limit], origin='lower', aspect='auto',
                    extent=[0, tg.shape[1] * step, .5 * step, (limit - .5) * step],
                    cmap='magma', vmin=0, vmax=1, interpolation='nearest', rasterized=True)
        axes[0, col].set(xlim=(0, duration), xlabel='Time (s)', ylabel='Delay (s)',
                         title=f'{title}\nLocal autocorrelation (lag 1-100 shown)')
        fig.colorbar(image, ax=axes[0, col], label='Normalized autocorrelation', shrink=.85)
        delay = np.arange(1, TEMPOGRAM_WIN) * step
        mean, std = tg[1:].mean(axis=1), tg[1:].std(axis=1)
        axes[1, col].fill_between(delay, np.maximum(0, mean - std), np.minimum(1, mean + std),
                                  color=BLUE, alpha=.2, label='Temporal mean +/- std')
        axes[1, col].plot(delay, mean, color=BLUE, linewidth=1.5)
        axes[1, col].set(xlim=(0, 3.84), ylim=(0, 1), xlabel='Delay (s)',
                         ylabel='Autocorrelation', title='Time pooling: 192 means + 192 stds')
        for delay_sec, bpm in ((.25, 240), (.5, 120), (1., 60)):
            axes[1, col].axvline(delay_sec, color='#7c8794', linestyle=':', linewidth=.8)
            axes[1, col].text(delay_sec + .025, .88, f'{bpm}', fontsize=8, color='#5d6670')
        axes[1, col].legend(frameon=False, loc='upper right')
    save(fig, '03_tempogram')


def structure_figure():
    import librosa
    fig = plt.figure(figsize=(12, 7.5), layout='constrained')
    grid = fig.add_gridspec(2, 2, height_ratios=(.55, 1.5))
    for col, (key, title) in enumerate(STRUCTURE_CASES):
        y, sr = audio_record(key)
        duration = len(y) / sr
        hop = max(1, round(sr * .05))
        raw = np.asarray(librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop), dtype=float)
        normalized = raw / np.maximum(np.linalg.norm(raw, axis=0, keepdims=True), 1e-8)
        pooled = _time_pool(normalized, MAX_FRAMES)
        ssm = pooled.T @ pooled  # Preserve original implementation; do not renormalize.
        novelty = _foote_novelty(ssm, NOVELTY_HALF)
        off_diag = ssm[~np.eye(len(ssm), dtype=bool)]
        ssm_mean = float(off_diag.mean())
        recurrence = float(np.mean(off_diag >= .75))
        grade = EVIDENCE['figure_audio'][key]['difficulty_grade']
        dt = duration / pooled.shape[1]
        time = (np.arange(pooled.shape[1]) + .5) * dt
        threshold = novelty.mean() + novelty.std()
        peaks = np.flatnonzero((novelty[1:-1] > novelty[:-2])
                    & (novelty[1:-1] > novelty[2:]) & (novelty[1:-1] > threshold)) + 1
        ax = fig.add_subplot(grid[0, col])
        ax.imshow(pooled, aspect='auto', origin='lower', extent=[0, duration, -.5, 11.5],
                  cmap='viridis', vmin=0, vmax=1, rasterized=True)
        ax.set(yticks=[0, 2, 4, 5, 7, 9, 11], yticklabels=['C', 'D', 'E', 'F', 'G', 'A', 'B'],
               ylabel='Pitch class', title=f'{title} (grade {grade})')
        ax = fig.add_subplot(grid[1, col])
        image = ax.imshow(ssm, origin='lower', extent=[0, duration, 0, duration],
                          vmin=0, vmax=1, cmap='viridis', rasterized=True)
        ax.set(xlabel='Time (s)', ylabel='Time (s)',
               title=f'SSM: mean = {ssm_mean:.3f}; fraction >= 0.75 = {recurrence:.1%}')
        fig.colorbar(image, ax=ax, label='Dot-product similarity', fraction=.047, pad=.02)
        EVIDENCE['figure_audio'][key]['structure_display'] = {
            'ssm_offdiagonal_mean': ssm_mean, 'recurrence_rate': recurrence,
            'novelty_mean': float(novelty.mean()), 'novelty_std': float(novelty.std()),
            'novelty_peak_density': len(peaks) / duration,
            'pooled_frames': pooled.shape[1], 'pooled_frame_seconds': dt,
            'novelty_half_window_seconds': NOVELTY_HALF * dt,
            'novelty_peak_times': time[peaks].tolist(),
            'no_normalization_after_pooling': True,
        }
        del y, raw, normalized
    save(fig, '04_structure_ssm')


def errors_figure(tables):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), layout='constrained',
                             gridspec_kw={'width_ratios': [1, 1.2]})
    grades = range(11)
    for model, label, color in zip(MODELS, LABELS, (BLUE, ORANGE, GREEN, RED)):
        table = tables[model]
        bias = [np.mean([r['score'] - r['true_grade'] for r in table.values()
                        if r['true_grade'] == grade]) for grade in grades]
        axes[0].plot(grades, bias, marker='o', markersize=4, color=color, label=label)
        EVIDENCE.setdefault('grade_bias', {})[model] = bias
    axes[0].axhline(0, color='#6d7781', linewidth=.8)
    axes[0].set(xticks=list(grades), xlabel='True difficulty grade',
                ylabel='Mean prediction - true grade', title='A  Grade-dependent bias (validation)')
    axes[0].legend(frameon=False, fontsize=9)
    cases = [
        (*RHYTHM_CASES[1], 'cqt_rhythm', '+ rhythm'),
        (*RHYTHM_CASES[0], 'cqt_rhythm', '+ rhythm'),
        (*STRUCTURE_CASES[0], 'cqt_structure', '+ structure'),
        (*STRUCTURE_CASES[1], 'cqt_structure', '+ structure'),
        ('Bailey K.Scoobie du wup', 'Bailey: Scoobie du wup', 'cqt_structure', '+ structure'),
        ('Copland A.Piano Variations', 'Copland: Piano Variations', 'cqt_structure', '+ structure'),
    ]
    labels = []
    for index, (key, label, model, addition) in enumerate(cases):
        baseline, added = tables['cqt'][key], tables[model][key]
        truth, p0, p1 = baseline['true_grade'], baseline['score'], added['score']
        axes[1].plot([p0, p1], [index, index], color='#aeb7bf', linewidth=2)
        for value, marker, color, legend in ((truth, 'D', '#1b2530', 'True grade'),
                (p0, 'o', BLUE, 'CQT'), (p1, 's', ORANGE, 'CQT + indicated feature')):
            axes[1].scatter(value, index, marker=marker, color=color, s=38,
                            label=legend if index == 0 else None, zorder=3)
        labels.append(f'{label}\n{addition}')
        EVIDENCE.setdefault('illustrative_cases', {})[key] = {
            'true_grade': truth, 'baseline_prediction': p0, 'added_model': model,
            'added_prediction': p1, 'absolute_error_change': abs(p1-truth) - abs(p0-truth)}
    axes[1].set(yticks=range(len(cases)), yticklabels=labels, xlim=(-.2, 10.5),
                xticks=range(0, 11, 2), xlabel='Predicted / true difficulty grade',
                title='B  Selected cases: gains and failures')
    axes[1].set_ylim(len(cases) - .6, -1.0)
    axes[1].legend(frameon=False, fontsize=8, loc='upper center', ncol=1)
    save(fig, '05_error_analysis')


if __name__ == '__main__':
    (OUT / 'figures').mkdir(parents=True, exist_ok=True)
    EVIDENCE['builder_sha256'] = sha256(Path(__file__))
    tables = validation()
    duration_figure()
    rhythm_figures()
    structure_figure()
    errors_figure(tables)
    EVIDENCE['audio_update_list_rows'] = sum(1 for _ in csv.DictReader(
        (ROOT/'datasets/audio/reports/audio_update_list.csv').open(encoding='utf-8-sig', newline='')))
    (OUT / 'report_evidence.json').write_text(json.dumps(EVIDENCE, ensure_ascii=False,
                                indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print('Finished five figures (PNG + PDF) and report_evidence.json.', flush=True)
