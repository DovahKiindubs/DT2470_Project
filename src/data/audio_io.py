"""原始音频加载入口：mono / 保留文件原生采样率 / 峰值归一化。

后续节奏特征请用时间单位（秒）设 hop/窗口。不落盘 WAV；MP3 为唯一原始音频源。
"""
from pathlib import Path

import numpy as np

PEAK_TARGET = 0.99
MIN_DURATION_SEC = 3.0
LOW_RMS_THRESHOLD = 1e-4
CLIP_FRACTION_WARN = 1e-3
LONG_DURATION_SEC = 600.0
DURATION_REL_ERR_WARN = 0.10


class AudioLoadError(Exception):
    """解码或校验失败。"""


def peak_normalize(y, peak_target=PEAK_TARGET):
    """将波形峰值缩放到 peak_target；全静音原样返回。"""
    y = np.asarray(y, dtype=np.float32)
    if y.ndim != 1:
        raise ValueError(f'Expected mono 1-D waveform, got shape {y.shape}')
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak == 0.0:
        return y.copy(), peak
    return (y * (peak_target / peak)).astype(np.float32, copy=False), peak


def load_audio(path, *, normalize=True, sr=None):
    """加载单声道 float32 波形；默认保留文件原生采样率（librosa sr=None）。

    Returns
    -------
    y : np.ndarray, shape (n_samples,), dtype float32
    sr_out : int
    """
    path = Path(path)
    if not path.is_file():
        raise AudioLoadError(f'Missing audio file: {path}')
    if path.is_symlink():
        raise AudioLoadError(f'Refusing symlink: {path}')

    try:
        import librosa
    except ImportError as exc:
        raise AudioLoadError('librosa is required to load audio') from exc

    try:
        y, sr_out = librosa.load(str(path), sr=sr, mono=True, dtype=np.float32)
    except Exception as exc:
        raise AudioLoadError(f'Decode failed for {path.name}: {exc}') from exc

    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if not np.isfinite(y).all():
        raise AudioLoadError(f'Non-finite samples in {path.name}')
    if normalize:
        y, _ = peak_normalize(y)
    return y, int(sr_out)


def waveform_stats(y, sr):
    """从已加载（通常为未归一化）波形提取 QC 统计。"""
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    n = int(y.size)
    duration = n / float(sr) if sr else 0.0
    if n == 0:
        return {
            'n_samples': 0,
            'duration_sec': 0.0,
            'peak': 0.0,
            'rms': 0.0,
            'clip_fraction': 0.0,
            'has_nan_inf': False,
        }
    abs_y = np.abs(y)
    peak = float(abs_y.max())
    rms = float(np.sqrt(np.mean(np.square(y), dtype=np.float64)))
    clip_fraction = float(np.mean(abs_y > PEAK_TARGET)) if peak > 0 else 0.0
    return {
        'n_samples': n,
        'duration_sec': float(duration),
        'peak': peak,
        'rms': rms,
        'clip_fraction': clip_fraction,
        'has_nan_inf': bool(not np.isfinite(y).all()),
    }


def qc_flags(stats, meta_duration=None):
    """硬失败：too_short / silence / non_finite；其余为软警告。"""
    hard, soft = [], []
    duration = float(stats['duration_sec'])
    peak = float(stats['peak'])
    rms = float(stats['rms'])

    if stats.get('has_nan_inf'):
        hard.append('non_finite')
    if duration < MIN_DURATION_SEC:
        hard.append('too_short')
    if peak == 0.0:
        hard.append('silence')

    if meta_duration is None:
        soft.append('missing_meta_duration')
    else:
        meta_duration = float(meta_duration)
        if meta_duration > 0:
            rel_err = abs(duration - meta_duration) / meta_duration
            if rel_err > DURATION_REL_ERR_WARN:
                soft.append('duration_mismatch')
        else:
            soft.append('missing_meta_duration')

    if peak > 0 and rms < LOW_RMS_THRESHOLD:
        soft.append('low_rms')
    if float(stats.get('clip_fraction', 0.0)) > CLIP_FRACTION_WARN:
        soft.append('clipping')
    if duration > LONG_DURATION_SEC:
        soft.append('long_duration')

    return hard, soft


def duration_rel_err(duration_sec, meta_duration):
    if meta_duration is None:
        return None
    meta_duration = float(meta_duration)
    if meta_duration <= 0:
        return None
    return float(abs(float(duration_sec) - meta_duration) / meta_duration)
