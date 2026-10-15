"""从 MP3 提取 chroma 自相似矩阵（SSM）结构统计；缓存格式与节奏特征同构。

为控制内存，先将 chroma 时间轴压到固定帧数再算 SSM；用对角 lag 剖面刻画重复。
"""
import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from data.audio_io import AudioLoadError, load_audio

ROOT = Path(__file__).resolve().parents[2]
HOP_SEC = 0.05
MAX_FRAMES = 256
LAG_MAX = 64
RECUR_THRESH = 0.75
NOVELTY_HALF = 8
SCALAR_NAMES = (
    'ssm_mean',
    'ssm_std',
    'recurrence_rate',
    'novelty_mean',
    'novelty_std',
    'novelty_peak_density',
    'lag_profile_mean',
    'lag_profile_std',
    'lag_profile_max',
)
FEATURE_NAMES = SCALAR_NAMES + tuple(f'ssm_lag_{j:02d}' for j in range(1, LAG_MAX + 1))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _hop_length(sr):
    return max(1, int(round(int(sr) * HOP_SEC)))


def _time_pool(chroma, max_frames):
    """(12, T) -> (12, max_frames)，过长则均分平均池化。"""
    chroma = np.asarray(chroma, dtype=np.float64)
    if chroma.ndim != 2 or chroma.shape[0] == 0 or chroma.shape[1] == 0:
        raise ValueError(f'Invalid chroma shape: {chroma.shape}')
    t = chroma.shape[1]
    if t <= max_frames:
        return chroma
    edges = np.linspace(0, t, max_frames + 1).astype(int)
    cols = []
    for i in range(max_frames):
        a, b = edges[i], edges[i + 1]
        cols.append(chroma[:, a:b].mean(axis=1) if b > a else chroma[:, min(a, t - 1)])
    return np.stack(cols, axis=1)


def _foote_novelty(ssm, half=NOVELTY_HALF):
    n = ssm.shape[0]
    novelty = np.zeros(n, dtype=np.float64)
    for i in range(n):
        a0, a1 = max(0, i - half), i
        b0, b1 = i, min(n, i + half)
        if a1 > a0 and b1 > b0:
            novelty[i] = -float(ssm[a0:a1, b0:b1].mean())
    novelty -= novelty.min()
    return novelty


def summarize_structure(y, sr):
    """chroma SSM 重复相关统计向量。"""
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
    duration = float(y.size) / float(sr)
    chroma = np.asarray(
        librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop), dtype=np.float64)
    norms = np.linalg.norm(chroma, axis=0, keepdims=True)
    chroma = chroma / np.maximum(norms, 1e-8)
    chroma = _time_pool(chroma, MAX_FRAMES)

    ssm = chroma.T @ chroma
    n = ssm.shape[0]
    if n < 2 or not np.isfinite(ssm).all():
        raise ValueError('Invalid SSM')

    off = ssm[np.triu_indices(n, k=1)]
    ssm_mean = float(off.mean())
    ssm_std = float(off.std(ddof=0))
    recurrence_rate = float(np.mean(off >= RECUR_THRESH))

    lags = np.zeros(LAG_MAX, dtype=np.float64)
    for lag in range(1, LAG_MAX + 1):
        if lag < n:
            lags[lag - 1] = float(np.mean(np.diag(ssm, k=lag)))

    novelty = _foote_novelty(ssm)
    novelty_mean = float(novelty.mean())
    novelty_std = float(novelty.std(ddof=0))
    if novelty.size >= 3 and novelty_std > 0:
        thr = novelty_mean + novelty_std
        peaks = int(np.sum(
            (novelty[1:-1] > novelty[:-2]) & (novelty[1:-1] > novelty[2:]) & (novelty[1:-1] > thr)))
        novelty_peak_density = float(peaks) / duration if duration > 0 else 0.0
    else:
        novelty_peak_density = 0.0

    scalars = np.asarray([
        ssm_mean,
        ssm_std,
        recurrence_rate,
        novelty_mean,
        novelty_std,
        novelty_peak_density,
        float(lags.mean()),
        float(lags.std(ddof=0)),
        float(lags.max()),
    ], dtype=np.float64)
    vec = np.concatenate((scalars, lags))
    if vec.shape != (len(FEATURE_NAMES),) or not np.isfinite(vec).all():
        raise ValueError('Structure summary must be finite and match FEATURE_NAMES')
    return vec


def _ok_keys_from_qc(qc_report):
    doc = json.loads(Path(qc_report).read_text(encoding='utf-8'))
    ok = doc.get('ok')
    if not isinstance(ok, list) or not ok:
        raise ValueError('QC report must contain a nonempty ok list')
    keys = []
    for row in ok:
        if not isinstance(row, dict) or 'key' not in row:
            raise ValueError('Each QC ok row must be an object with key')
        keys.append(row['key'])
    return sorted(keys)


def _extract_one(item):
    key, path, grade = item
    try:
        y, sr = load_audio(path, normalize=True)
        feat = summarize_structure(y, sr)
        return key, feat, int(grade), sha256(path)
    except (AudioLoadError, ValueError, OSError) as exc:
        raise ValueError(f'Structure extraction failed for {key}: {exc}') from exc


def _emit(log_fp, message):
    print(message, flush=True)
    if log_fp is not None:
        log_fp.write(message + '\n')
        log_fp.flush()


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
        orphans = sorted(set(files) - set(metadata))
        if orphans:
            raise ValueError(f'Orphan audio not in metadata: {len(orphans)}; pass qc_report or keys')

    missing_meta = [k for k in ids if k not in metadata]
    missing_audio = [k for k in ids if k not in files]
    if missing_meta or missing_audio:
        raise ValueError(
            f'Key mismatch vs requested set: {len(missing_meta)} missing metadata, '
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
        _emit(log_fp, f'=== structure extract start {datetime.now(timezone.utc).isoformat()} ===')
        _emit(log_fp, f'n={len(ids)} workers={workers} output={output}')
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
        _emit(log_fp, f'Saved {len(ids)} recordings x {len(FEATURE_NAMES)} features: {output}  '
              f'total={total:.1f}s')
        _emit(log_fp, f'=== structure extract end {datetime.now(timezone.utc).isoformat()} ===')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio-dir', type=Path, default=ROOT / 'datasets/audio/mp3_pieces')
    parser.add_argument('--metadata', type=Path, default=ROOT / 'datasets/metadata/new_clean_data.json')
    parser.add_argument('--qc-report', type=Path,
                        default=ROOT / 'datasets/audio/reports/audio_qc.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'datasets/features/structure_stats.npz')
    parser.add_argument('--log', type=Path, default=None)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    extract(args.audio_dir, args.metadata, args.output, qc_report=args.qc_report,
            workers=args.workers, log_path=args.log)
