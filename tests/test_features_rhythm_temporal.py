"""时序节奏特征测试。"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from features_rhythm_temporal import FEATURE_NAMES, extract, summarize_rhythm_temporal


def _tone(sr=22050, seconds=12.0, hz=440.0):
    t = np.arange(int(sr * seconds), dtype=np.float32) / sr
    # 加少量脉冲，避免 onset 全空
    y = 0.2 * np.sin(2 * np.pi * hz * t)
    for k in range(0, int(seconds), 1):
        i0 = int(k * sr)
        y[i0:i0 + int(0.01 * sr)] += 0.5
    return y.astype(np.float32), sr


def test_summarize_shape_finite():
    vec = summarize_rhythm_temporal(*_tone())
    assert vec.shape == (len(FEATURE_NAMES),) == (36,)
    assert list(FEATURE_NAMES[:9]) == [f'mean_{n}' for n in (
        'onset_density', 'onset_strength_mean', 'onset_strength_std', 'onset_strength_max',
        'ioi_mean', 'ioi_std', 'ioi_cv', 'tempo_bpm', 'tempo_confidence')]
    assert np.isfinite(vec).all()


def test_rejects_bad():
    with pytest.raises(ValueError):
        summarize_rhythm_temporal(np.zeros(0, np.float32), 22050)


def test_extract(tmp_path, monkeypatch):
    audio = tmp_path / 'mp3'
    audio.mkdir()
    for name in ('a', 'b'):
        (audio / f'{name}.mp3').write_bytes(b'x')
    metadata = tmp_path / 'm.json'
    metadata.write_text(json.dumps({'a': {'ps': '1'}, 'b': {'ps': '2'}}), encoding='utf-8')
    qc = tmp_path / 'qc.json'
    qc.write_text(json.dumps({'ok': [{'key': 'a'}, {'key': 'b'}]}), encoding='utf-8')
    monkeypatch.setattr('features_rhythm_temporal.load_audio', lambda path, normalize=True: _tone())
    out = tmp_path / 'rhythm_temporal.npz'
    extract(audio, metadata, out, qc_report=qc)
    with np.load(out, allow_pickle=False) as d:
        assert d['X'].shape == (2, 36)
        assert d['ids'].tolist() == ['a', 'b']
    with pytest.raises(FileExistsError):
        extract(audio, metadata, out, qc_report=qc)
