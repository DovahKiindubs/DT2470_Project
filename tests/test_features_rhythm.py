"""节奏特征提取测试；不依赖正式 MP3 目录。"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from features_rhythm import FEATURE_NAMES, extract, summarize_rhythm


def _tone(sr=22050, seconds=4.0, hz=440.0):
    t = np.arange(int(sr * seconds), dtype=np.float32) / sr
    return (0.25 * np.sin(2 * np.pi * hz * t)).astype(np.float32), sr


def test_summarize_rhythm_shape_and_finite():
    y, sr = _tone()
    vec = summarize_rhythm(y, sr)
    assert vec.shape == (len(FEATURE_NAMES),)
    assert vec.dtype == np.float64
    assert np.isfinite(vec).all()
    assert vec[0] >= 0.0  # onset_density


def test_summarize_rhythm_rejects_bad_input():
    with pytest.raises(ValueError):
        summarize_rhythm(np.zeros(0, dtype=np.float32), 22050)
    with pytest.raises(ValueError):
        summarize_rhythm(np.array([np.nan], dtype=np.float32), 22050)
    with pytest.raises(ValueError):
        summarize_rhythm(np.ones(1000, dtype=np.float32), 0)


def test_extract_alignment_qc_and_no_overwrite(tmp_path, monkeypatch):
    audio = tmp_path / 'mp3'
    audio.mkdir()
    for name in ('z', 'a'):
        (audio / f'{name}.mp3').write_bytes(b'fake')

    metadata = tmp_path / 'metadata.json'
    metadata.write_text(json.dumps({'z': {'ps': '2'}, 'a': {'ps': '1'}, 'm': {'ps': '3'}}),
                        encoding='utf-8')
    qc = tmp_path / 'qc.json'
    qc.write_text(json.dumps({'ok': [{'key': 'z', 'flags': []}, {'key': 'a', 'flags': []}]}),
                  encoding='utf-8')

    def fake_load(path, normalize=True):
        return _tone()

    monkeypatch.setattr('features_rhythm.load_audio', fake_load)

    output = tmp_path / 'rhythm_stats.npz'
    log_path = tmp_path / 'rhythm_stats_extract.log'
    extract(audio, metadata, output, qc_report=qc, log_path=log_path)
    with np.load(output, allow_pickle=False) as d:
        assert set(d.files) == {
            'X', 'y', 'ids', 'feature_names', 'audio_sha256', 'metadata_sha256',
        }
        assert d['ids'].tolist() == ['a', 'z']
        assert d['y'].tolist() == [1, 2]
        assert d['X'].shape == (2, len(FEATURE_NAMES))
        assert d['feature_names'].tolist() == list(FEATURE_NAMES)
        assert d['audio_sha256'].shape == (2,)
        assert d['metadata_sha256'].shape == ()
    text = log_path.read_text(encoding='utf-8')
    assert 'rhythm extract start' in text and 'Saved 2 recordings' in text

    with pytest.raises(FileExistsError):
        extract(audio, metadata, output, qc_report=qc)


def test_extract_no_silent_key_drop(tmp_path, monkeypatch):
    audio = tmp_path / 'mp3'
    audio.mkdir()
    (audio / 'a.mp3').write_bytes(b'fake')
    metadata = tmp_path / 'metadata.json'
    metadata.write_text(json.dumps({'a': {'ps': '1'}}), encoding='utf-8')
    qc = tmp_path / 'qc.json'
    qc.write_text(json.dumps({'ok': [{'key': 'a'}, {'key': 'missing'}]}), encoding='utf-8')
    monkeypatch.setattr('features_rhythm.load_audio', lambda path, normalize=True: _tone())
    with pytest.raises(ValueError, match='missing audio'):
        extract(audio, metadata, tmp_path / 'out.npz', qc_report=qc)
