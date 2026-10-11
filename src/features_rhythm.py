"""从 MP3 提取 onset / tempogram 节奏统计；缓存格式与 cqt_stats.npz 同构。

hop/窗用秒换算，适配原生采样率。不落盘中间 WAV，不修改原始 MP3。
CPU 密集步骤可用多进程加速（--workers）；多线程受 GIL 限制收益很小。
"""
import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from audio_io import AudioLoadError, load_audio

ROOT = Path(__file__).resolve().parents[1]
HOP_SEC = 0.02
FEATURE_NAMES = (
    'onset_density',
    'onset_strength_mean',
    'onset_strength_std',
    'onset_strength_p95',
    'ioi_mean',
    'ioi_std',
    'tempo_bpm',
    'tempogram_mean',
    'tempogram_std',
    'tempogram_max',
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _hop_length(sr):
    return max(1, int(round(int(sr) * HOP_SEC)))


def summarize_rhythm(y, sr):
    """单曲节奏统计向量，长度与 FEATURE_NAMES 一致。"""
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
    onset_env = np.asarray(
        librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop), dtype=np.float64)
    if onset_env.size == 0 or not np.isfinite(onset_env).all():
        raise ValueError('Invalid onset strength envelope')

    onset_times = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, hop_length=hop, units='time')
    onset_times = np.asarray(onset_times, dtype=np.float64)
    onset_density = float(onset_times.size) / duration if duration > 0 else 0.0

    if onset_times.size >= 2:
        ioi = np.diff(onset_times)
        ioi_mean = float(ioi.mean())
        ioi_std = float(ioi.std(ddof=0))
    else:
        ioi_mean, ioi_std = 0.0, 0.0

    tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr, hop_length=hop)
    tempo_bpm = float(np.asarray(tempo, dtype=np.float64).reshape(-1)[0])

    tempogram = np.asarray(
        librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=hop),
        dtype=np.float64)
    if tempogram.size == 0 or not np.isfinite(tempogram).all():
        raise ValueError('Invalid tempogram')

    vec = np.asarray([
        onset_density,
        float(onset_env.mean()),
        float(onset_env.std(ddof=0)),
        float(np.percentile(onset_env, 95)),
        ioi_mean,
        ioi_std,
        tempo_bpm,
        float(tempogram.mean()),
        float(tempogram.std(ddof=0)),
        float(tempogram.max()),
    ], dtype=np.float64)
    if vec.shape != (len(FEATURE_NAMES),) or not np.isfinite(vec).all():
        raise ValueError('Rhythm summary must be finite and match FEATURE_NAMES')
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
    """进程池 worker：返回 (key, feature, grade, audio_sha256)。"""
    key, path, grade = item
    try:
        y, sr = load_audio(path, normalize=True)
        feat = summarize_rhythm(y, sr)
        return key, feat, int(grade), sha256(path)
    except (AudioLoadError, ValueError, OSError) as exc:
        raise ValueError(f'Rhythm extraction failed for {key}: {exc}') from exc


def _emit(log_fp, message):
    print(message, flush=True)
    if log_fp is not None:
        log_fp.write(message + '\n')
        log_fp.flush()


def extract(audio_dir, metadata_path, output, *, qc_report=None, keys=None,
            workers=1, log_path=None):
    """写出 rhythm_stats.npz：X,y,ids,feature_names,audio_sha256,metadata_sha256。"""
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
        _emit(log_fp, f'=== rhythm extract start {datetime.now(timezone.utc).isoformat()} ===')
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
                key, feat, grade, digest = _extract_one(item)
                _on_result(key, feat, grade, digest)
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_extract_one, item): item[0] for item in jobs}
                for fut in as_completed(futures):
                    key, feat, grade, digest = fut.result()
                    _on_result(key, feat, grade, digest)

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
        _emit(log_fp, f'=== rhythm extract end {datetime.now(timezone.utc).isoformat()} ===')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio-dir', type=Path, default=ROOT / 'datasets/audio/mp3_pieces')
    parser.add_argument('--metadata', type=Path, default=ROOT / 'datasets/metadata/new_clean_data.json')
    parser.add_argument('--qc-report', type=Path,
                        default=ROOT / 'datasets/audio/reports/audio_qc.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'datasets/features/rhythm_stats.npz')
    parser.add_argument('--log', type=Path, default=None,
                        help='默认写到 <output_stem>_extract.log')
    parser.add_argument('--workers', type=int, default=1,
                        help='进程数；>1 用多进程。多线程对 librosa 加速很有限')
    args = parser.parse_args()
    extract(args.audio_dir, args.metadata, args.output, qc_report=args.qc_report,
            workers=args.workers, log_path=args.log)
