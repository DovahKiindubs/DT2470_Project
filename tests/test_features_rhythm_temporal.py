"""时序节奏窗序列测试。"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from features_rhythm_temporal import FEATURE_NAMES, WIN_SEC, HOP_WIN_SEC, compute_rhythm_windows, extract


def _tone(sr=22050, seconds=12.0, hz=440.0):
    t = np.arange(int(sr * seconds), dtype=np.float32) / sr
    y = 0.2 * np.sin(2 * np.pi * hz * t)
    for k in range(0, int(seconds), 1):
        i0 = int(k * sr)
        y[i0:i0 + int(0.01 * sr)] += 0.5
    return y.astype(np.float32), sr


def test_compute_shape_and_times():
    X, times = compute_rhythm_windows(*_tone())
    assert X.shape[0] == len(FEATURE_NAMES) == 9
    assert X.shape[1] == times.shape[0] >= 2
    assert np.isfinite(X).all() and np.isfinite(times).all()
    assert times[0] == pytest.approx(0.0, abs=1e-6)
    assert np.all(np.diff(times) >= 0)
    assert WIN_SEC == 4.0 and HOP_WIN_SEC == 0.2


def test_rejects_bad():
    with pytest.raises(ValueError):
        compute_rhythm_windows(np.zeros(0, np.float32), 22050)


def test_extract_and_resume(tmp_path, monkeypatch):
    audio = tmp_path / 'mp3'
    audio.mkdir()
    for name in ('a', 'b'):
        (audio / f'{name}.mp3').write_bytes(b'x')
    metadata = tmp_path / 'm.json'
    metadata.write_text(json.dumps({'a': {'ps': '1'}, 'b': {'ps': '2'}}), encoding='utf-8')
    qc = tmp_path / 'qc.json'
    qc.write_text(json.dumps({'ok': [{'key': 'a'}, {'key': 'b'}]}), encoding='utf-8')
    monkeypatch.setattr('features_rhythm_temporal.load_audio', lambda path, normalize=True: _tone())
    out = tmp_path / 'rhythm_windows'
    extract(audio, metadata, out, qc_report=qc)
    with np.load(out / 'index.npz', allow_pickle=False) as idx:
        assert idx['ids'].tolist() == ['a', 'b']
    # resume with nothing to do should rewrite index
    extract(audio, metadata, out, qc_report=qc, resume=True)
    assert (out / 'a.npz').exists()
    with pytest.raises(FileExistsError):
        extract(audio, metadata, out, qc_report=qc)
