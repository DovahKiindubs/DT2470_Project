"""滑窗局部节奏序列（供 CRNN 等序列模型）。

全曲只算一次 onset 包络并做一次 onset 检测，再按时间滑窗取局部统计。
默认 4s 窗 / 0.2s 步；逐曲保存 X=(9,N)、times=(N,) 起点秒。
支持 --resume：跳过已有 <id>.npz，全部完成后写 index.npz。
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from data.audio_io import AudioLoadError, load_audio
from features_rhythm import (
    HOP_SEC, ROOT, TEMPOGRAM_WIN, _emit, _hop_length, _ok_keys_from_qc, sha256,
)

WIN_SEC = 4.0
HOP_WIN_SEC = 0.2
FEATURE_NAMES = (
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


def _window_row(seg, onset_times, t0, t1, sr, hop):
    """onset/IOI 用全曲检测结果切片；tempo 与 confidence 与全曲版同定义（逐窗）。"""
    import librosa

    duration = t1 - t0
    if seg.size == 0 or duration <= 0 or not np.isfinite(seg).all():
        return np.zeros(len(FEATURE_NAMES), dtype=np.float64)

    mask = (onset_times >= t0) & (onset_times < t1)
    local = onset_times[mask]
    dens = float(local.size) / duration

    if local.size >= 2:
        ioi = np.diff(local)
        ioi_mean = float(ioi.mean())
        ioi_std = float(ioi.std(ddof=0))
        ioi_cv = ioi_std / ioi_mean if ioi_mean > 0 else 0.0
    else:
        ioi_mean = ioi_std = ioi_cv = 0.0

    # 与 features_rhythm.py 一致：librosa.tempo + tempogram 剖面峰均比
    tempo = librosa.feature.tempo(onset_envelope=seg, sr=sr, hop_length=hop)
    tempo_bpm = float(np.asarray(tempo, dtype=np.float64).reshape(-1)[0])
    tg_win = min(TEMPOGRAM_WIN, max(2, int(seg.size)))
    tempogram = np.asarray(
        librosa.feature.tempogram(
            onset_envelope=seg, sr=sr, hop_length=hop, win_length=tg_win),
        dtype=np.float64)
    tg_profile = tempogram.mean(axis=1)
    tempo_conf = float(tg_profile.max() / (tg_profile.mean() + 1e-8))

    return np.asarray([
        dens,
        float(seg.mean()), float(seg.std(ddof=0)), float(seg.max()),
        ioi_mean, ioi_std, ioi_cv,
        tempo_bpm, tempo_conf,
    ], dtype=np.float64)


def compute_rhythm_windows(y, sr, win_sec=WIN_SEC, hop_sec=HOP_WIN_SEC):
    """返回 X (9, N)、times (N,) 窗起点秒。通道维在前，便于 Conv1d/CRNN。"""
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if y.size == 0:
        raise ValueError('Empty waveform')
    if not np.isfinite(y).all():
        raise ValueError('Non-finite samples')
    sr = int(sr)
    if sr <= 0:
        raise ValueError(f'Invalid sample rate: {sr}')

    import librosa

    hop = _hop_length(sr)
    onset_env = np.asarray(
        librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop), dtype=np.float64)
    if onset_env.size == 0 or not np.isfinite(onset_env).all():
        raise ValueError('Invalid onset strength envelope')

    onset_times = np.asarray(
        librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=sr, hop_length=hop, units='time'),
        dtype=np.float64)

    win = max(1, int(round(win_sec / HOP_SEC)))
    hop_w = max(1, int(round(hop_sec / HOP_SEC)))
    rows, times = [], []
    for start in _window_starts(onset_env.size, win, hop_w):
        end = min(start + win, onset_env.size)
        t0 = float(start * hop) / float(sr)
        t1 = float(end * hop) / float(sr)
        rows.append(_window_row(onset_env[start:end], onset_times, t0, t1, sr, hop))
        times.append(t0)
    X = np.asarray(rows, dtype=np.float64).T  # (9, N)
    times = np.asarray(times, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] != len(FEATURE_NAMES) or X.shape[1] == 0:
        raise ValueError(f'Expected (9, N>0), got {X.shape}')
    if times.shape != (X.shape[1],) or not np.isfinite(X).all() or not np.isfinite(times).all():
        raise ValueError('Invalid window sequence or timestamps')
    return X, times


def _extract_one(item):
    key, path, grade, out_path = item
    try:
        y, sr = load_audio(path, normalize=True)
        try:
            X, times = compute_rhythm_windows(y, sr)
        except MemoryError:
            # 超长曲目在多进程下可能 OOM：释放后再试一次
            import gc
            gc.collect()
            X, times = compute_rhythm_windows(y, sr)
        return key, X, times, int(grade), sha256(path), str(out_path)
    except (AudioLoadError, ValueError, OSError, MemoryError) as exc:
        raise ValueError(f'Temporal rhythm extraction failed for {key}: {exc}') from exc


def extract(audio_dir, metadata_path, output_dir, *, qc_report=None, keys=None,
            workers=1, log_path=None, resume=False):
    """写入 output_dir/<id>.npz（X,times）与 index.npz（ids,y,lengths,…）。"""
    audio_dir, metadata_path, output_dir = map(Path, (audio_dir, metadata_path, output_dir))
    workers = int(workers)
    if workers < 1:
        raise ValueError('workers must be >= 1')
    index_path = output_dir / 'index.npz'
    existing = set()
    if output_dir.exists():
        existing = {p.stem for p in output_dir.glob('*.npz')}
        if existing and not resume:
            raise FileExistsError(
                f'Refusing overwrite existing rhythm windows: {output_dir}; pass resume=True')
        if index_path.exists() and resume:
            index_path.unlink()
    else:
        output_dir.mkdir(parents=True, exist_ok=False)

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
        if key in existing:
            continue
        jobs.append((key, str(files[key]), int(raw_grade), str(output_dir / f'{key}.npz')))

    if log_path is None:
        log_path = output_dir / 'extract.log'
    else:
        log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    by_key = {}
    # 已完成的曲子：从现有 npz 回填 index 信息
    for key in ids:
        if key not in existing:
            continue
        path = output_dir / f'{key}.npz'
        with np.load(path, allow_pickle=False) as d:
            by_key[key] = (int(d['X'].shape[1]), int(d['y']), str(d['audio_sha256']))

    t0 = time.perf_counter()
    with log_path.open('a', encoding='utf-8') as log_fp:
        _emit(log_fp, f'=== rhythm windows extract start {datetime.now(timezone.utc).isoformat()} ===')
        _emit(log_fp, f'n={len(ids)} todo={len(jobs)} skipped={len(existing)} workers={workers} '
              f'win={WIN_SEC}s hop={HOP_WIN_SEC}s dim={len(FEATURE_NAMES)} output={output_dir}')
        done0 = len(by_key)
        done = 0

        def _on_result(key, X, times, grade, digest, out_path):
            nonlocal done
            out = Path(out_path)
            if out.exists():
                # resume 竞态或上次崩溃残留：沿用已有文件
                with np.load(out, allow_pickle=False) as d:
                    by_key[key] = (int(d['X'].shape[1]), int(d['y']), str(d['audio_sha256']))
            else:
                tmp = out.with_suffix('.npz.tmp')
                with tmp.open('wb') as f:
                    np.savez_compressed(
                        f, X=X, times=times, y=np.asarray(grade, dtype=np.int64),
                        feature_names=np.asarray(FEATURE_NAMES),
                        win_sec=np.asarray(WIN_SEC), hop_sec=np.asarray(HOP_WIN_SEC),
                        audio_sha256=np.asarray(digest))
                tmp.replace(out)
                by_key[key] = (X.shape[1], grade, digest)
            done += 1
            finished = done0 + done
            if done % 10 == 0 or done == len(jobs):
                elapsed = time.perf_counter() - t0
                rate = done / elapsed if elapsed > 0 else 0.0
                eta = (len(jobs) - done) / rate if rate > 0 else float('inf')
                _emit(log_fp, f'Extracted {finished}/{len(ids)} (+{done} this run)  '
                      f'elapsed={elapsed:.1f}s  {rate:.2f}/s  eta={eta:.1f}s')

        if not jobs:
            _emit(log_fp, 'Nothing to do; all pieces already extracted')
        elif workers == 1:
            for item in jobs:
                _on_result(*_extract_one(item))
        else:
            # 分批提交，避免一次塞满队列；也不要把 stdout/stderr 接到管道
            # （mpg123 警告填满管道会导致 worker 卡死）。
            batch = max(workers * 4, 32)
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for i in range(0, len(jobs), batch):
                    chunk = jobs[i:i + batch]
                    futures = {pool.submit(_extract_one, item): item[0] for item in chunk}
                    for fut in as_completed(futures):
                        _on_result(*fut.result())

        missing = [k for k in ids if k not in by_key]
        if missing:
            raise RuntimeError(f'Missing {len(missing)} outputs after extract, e.g. {missing[:3]}')

        lengths = np.asarray([by_key[k][0] for k in ids], dtype=np.int64)
        labels = np.asarray([by_key[k][1] for k in ids], dtype=np.int64)
        source_hashes = np.asarray([by_key[k][2] for k in ids])
        with index_path.open('xb') as f:
            np.savez_compressed(
                f,
                ids=np.asarray(ids),
                y=labels,
                lengths=lengths,
                feature_names=np.asarray(FEATURE_NAMES),
                audio_sha256=source_hashes,
                metadata_sha256=np.asarray(sha256(metadata_path)),
                win_sec=np.asarray(WIN_SEC),
                hop_sec=np.asarray(HOP_WIN_SEC),
            )
        total = time.perf_counter() - t0
        _emit(log_fp, f'Saved {len(ids)} sequences under {output_dir}  '
              f'lengths=[{lengths.min()},{lengths.max()}] total={total:.1f}s')
        _emit(log_fp, f'=== rhythm windows extract end {datetime.now(timezone.utc).isoformat()} ===')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audio-dir', type=Path, default=ROOT / 'datasets/audio/mp3_pieces')
    p.add_argument('--metadata', type=Path, default=ROOT / 'datasets/metadata/new_clean_data.json')
    p.add_argument('--qc-report', type=Path,
                   default=ROOT / 'datasets/audio/reports/audio_qc.json')
    p.add_argument('--output', type=Path, default=ROOT / 'datasets/features/rhythm_windows')
    p.add_argument('--log', type=Path, default=None)
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--resume', action='store_true',
                   help='Skip existing <id>.npz and rewrite index.npz at the end')
    args = p.parse_args()
    extract(args.audio_dir, args.metadata, args.output, qc_report=args.qc_report,
            workers=args.workers, log_path=args.log, resume=args.resume)
