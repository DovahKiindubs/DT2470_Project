"""滑窗局部节奏统计（时序版）；缓存格式与 rhythm_stats.npz 同构。

全曲只算一次 onset 包络，再在时间轴上滑窗；避免每窗重解码。
池化：全体窗 mean/std/max + 按 onset_density 取 top hard_fraction 的 hard_mean。
"""
import argparse
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from audio_io import AudioLoadError, load_audio
from features_rhythm import (
    HOP_SEC, ROOT, _emit, _hop_length, _ok_keys_from_qc, sha256,
)

WIN_SEC = 10.0
HOP_WIN_SEC = 5.0
HARD_FRACTION = 0.2
LOCAL_NAMES = (
    'onset_density',
    'onset_strength_mean',
    'onset_strength_std',
    'onset_strength_max',
    'ioi_mean',
    'ioi_std',
    'ioi_cv',
    'tempo_bpm',
    'tempo_confidence',
)
POOLS = ('mean', 'std', 'max', 'hard_mean')
FEATURE_NAMES = tuple(f'{pool}_{name}' for pool in POOLS for name in LOCAL_NAMES)


def _window_starts(n, win, hop):
    if n <= 0 or win <= 0 or hop <= 0:
        raise ValueError('Invalid window sizes')
    if n <= win:
        return [0]
    starts = list(range(0, n - win + 1, hop))
    last = n - win
    if starts[-1] != last:
        starts.append(last)
    return starts


def _local_from_onset(onset_env, sr, hop, start, end):
    """单窗局部节奏标量（在 onset 帧上切片）。"""
    import librosa

    seg = np.asarray(onset_env[start:end], dtype=np.float64)
    if seg.size == 0 or not np.isfinite(seg).all():
        return np.zeros(len(LOCAL_NAMES), dtype=np.float64)

    duration = float(seg.size * hop) / float(sr)
    times = librosa.onset.onset_detect(
        onset_envelope=seg, sr=sr, hop_length=hop, units='time')
    times = np.asarray(times, dtype=np.float64)
    dens = float(times.size) / duration if duration > 0 else 0.0

    if times.size >= 2:
        ioi = np.diff(times)
        ioi_mean = float(ioi.mean())
        ioi_std = float(ioi.std(ddof=0))
        ioi_cv = ioi_std / ioi_mean if ioi_mean > 0 else 0.0
    else:
        ioi_mean = ioi_std = ioi_cv = 0.0

    tempo = librosa.feature.tempo(onset_envelope=seg, sr=sr, hop_length=hop)
    tempo_bpm = float(np.asarray(tempo, dtype=np.float64).reshape(-1)[0])
    tempo_conf = float(seg.max() / (seg.mean() + 1e-8))

    return np.asarray([
        dens,
        float(seg.mean()), float(seg.std(ddof=0)), float(seg.max()),
        ioi_mean, ioi_std, ioi_cv,
        tempo_bpm, tempo_conf,
    ], dtype=np.float64)


def summarize_rhythm_temporal(y, sr, win_sec=WIN_SEC, hop_sec=HOP_WIN_SEC,
                              hard_fraction=HARD_FRACTION):
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if y.size == 0:
        raise ValueError('Empty waveform')
    if not np.isfinite(y).all():
        raise ValueError('Non-finite samples')
    sr = int(sr)
    if sr <= 0:
        raise ValueError(f'Invalid sample rate: {sr}')
    if not 0 < hard_fraction <= 1:
        raise ValueError('hard_fraction must be in (0, 1]')

    import librosa

    hop = _hop_length(sr)
    onset_env = np.asarray(
        librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop), dtype=np.float64)
    if onset_env.size == 0 or not np.isfinite(onset_env).all():
        raise ValueError('Invalid onset strength envelope')

    win = max(1, int(round(win_sec / HOP_SEC)))
    hop_w = max(1, int(round(hop_sec / HOP_SEC)))
    rows, scores = [], []
    for start in _window_starts(onset_env.size, win, hop_w):
        end = min(start + win, onset_env.size)
        local = _local_from_onset(onset_env, sr, hop, start, end)
        rows.append(local)
        scores.append(float(local[0]))  # onset_density 作难度代理
    rows = np.asarray(rows, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    n_hard = max(1, math.ceil(len(scores) * hard_fraction))
    hard_ix = np.argsort(scores, kind='stable')[-n_hard:]
    hard = rows[hard_ix]

    vec = np.concatenate((
        rows.mean(axis=0),
        rows.std(axis=0, ddof=0),
        rows.max(axis=0),
        hard.mean(axis=0),
    ))
    if vec.shape != (len(FEATURE_NAMES),) or not np.isfinite(vec).all():
        raise ValueError('Temporal rhythm summary must be finite and match FEATURE_NAMES')
    return vec


def _extract_one(item):
    key, path, grade = item
    try:
        y, sr = load_audio(path, normalize=True)
        feat = summarize_rhythm_temporal(y, sr)
        return key, feat, int(grade), sha256(path)
    except (AudioLoadError, ValueError, OSError) as exc:
        raise ValueError(f'Temporal rhythm extraction failed for {key}: {exc}') from exc


def extract(audio_dir, metadata_path, output, *, qc_report=None, keys=None,
            workers=1, log_path=None):
    audio_dir, metadata_path, output = map(Path, (audio_dir, metadata_path, output))
    workers = int(workers)
    if workers < 1:
        raise ValueError('workers must be >= 1')
    if output.exists():
        raise FileExistsError(f'Feature cache already exists: {output}')
    if output.suffix != '.npz':
        raise ValueError('Feature output must end in .npz')

    import json

    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    if not isinstance(metadata, dict) or not metadata:
        raise ValueError('Metadata must be a nonempty object')

    files = {p.stem: p for p in audio_dir.glob('*.mp3') if not p.is_symlink()}
    if keys is not None:
        ids = sorted(keys)
    elif qc_report is not None:
        ids = _ok_keys_from_qc(qc_report)
    else:
        ids = sorted(set(files) & set(metadata))

    missing_meta = [k for k in ids if k not in metadata]
    missing_audio = [k for k in ids if k not in files]
    if missing_meta or missing_audio:
        raise ValueError(
            f'Key mismatch: {len(missing_meta)} missing metadata, '
            f'{len(missing_audio)} missing audio; no silent exclusions')

    jobs = []
    for key in ids:
        raw_grade = metadata[key].get('ps')
        if str(raw_grade) not in {str(n) for n in range(11)}:
            raise ValueError(f'Invalid grade for {key}: {raw_grade}')
        jobs.append((key, str(files[key]), int(raw_grade)))

    if log_path is None:
        log_path = output.with_name(output.stem + '_extract.log')
    else:
        log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    by_key = {}
    t0 = time.perf_counter()
    with log_path.open('a', encoding='utf-8') as log_fp:
        _emit(log_fp, f'=== rhythm temporal extract start {datetime.now(timezone.utc).isoformat()} ===')
        _emit(log_fp, f'n={len(ids)} workers={workers} win={WIN_SEC}s hop={HOP_WIN_SEC}s '
              f'hard={HARD_FRACTION} dim={len(FEATURE_NAMES)} output={output}')
        done = 0

        def _on_result(key, feat, grade, digest):
            nonlocal done
            by_key[key] = (feat, grade, digest)
            done += 1
            if done % 50 == 0 or done == len(ids):
                elapsed = time.perf_counter() - t0
                rate = done / elapsed if elapsed > 0 else 0.0
                eta = (len(ids) - done) / rate if rate > 0 else float('inf')
                _emit(log_fp, f'Extracted {done}/{len(ids)}  elapsed={elapsed:.1f}s  '
                      f'{rate:.2f}/s  eta={eta:.1f}s')

        if workers == 1:
            for item in jobs:
                _on_result(*_extract_one(item))
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_extract_one, item): item[0] for item in jobs}
                for fut in as_completed(futures):
                    _on_result(*fut.result())

        features = [by_key[k][0] for k in ids]
        labels = [by_key[k][1] for k in ids]
        source_hashes = [by_key[k][2] for k in ids]
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('xb') as f:
            np.savez_compressed(
                f,
                X=np.asarray(features, dtype=np.float64),
                y=np.asarray(labels, dtype=np.int64),
                ids=np.asarray(ids),
                feature_names=np.asarray(FEATURE_NAMES),
                audio_sha256=np.asarray(source_hashes),
                metadata_sha256=np.asarray(sha256(metadata_path)),
            )
        total = time.perf_counter() - t0
        _emit(log_fp, f'Saved {len(ids)} x {len(FEATURE_NAMES)}: {output}  total={total:.1f}s')
        _emit(log_fp, f'=== rhythm temporal extract end {datetime.now(timezone.utc).isoformat()} ===')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audio-dir', type=Path, default=ROOT / 'datasets/audio/mp3_pieces')
    p.add_argument('--metadata', type=Path, default=ROOT / 'datasets/metadata/new_clean_data.json')
    p.add_argument('--qc-report', type=Path,
                   default=ROOT / 'datasets/audio/reports/audio_qc.json')
    p.add_argument('--output', type=Path, default=ROOT / 'datasets/features/rhythm_temporal.npz')
    p.add_argument('--log', type=Path, default=None)
    p.add_argument('--workers', type=int, default=1)
    args = p.parse_args()
    extract(args.audio_dir, args.metadata, args.output, qc_report=args.qc_report,
            workers=args.workers, log_path=args.log)
