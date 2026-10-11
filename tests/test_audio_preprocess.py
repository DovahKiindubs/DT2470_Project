"""音频加载与预处理质检测试；使用临时合成 WAV，不依赖正式 MP3 目录。"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from audio_io import (
    PEAK_TARGET,
    AudioLoadError,
    duration_rel_err,
    load_audio,
    peak_normalize,
    qc_flags,
    waveform_stats,
)
from preprocess_audio import (
    assemble_report,
    build_common_subset,
    qc_one,
    run_qc,
)

TEST_SR = 44100


def _write_wav(path, y, sr=TEST_SR):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y.astype(np.float32), sr)


def test_peak_normalize_and_silence():
    y = np.array([0.0, 0.5, -1.0], dtype=np.float32)
    out, peak = peak_normalize(y)
    assert peak == pytest.approx(1.0)
    assert float(np.max(np.abs(out))) == pytest.approx(PEAK_TARGET)
    silent = np.zeros(100, dtype=np.float32)
    out_s, peak_s = peak_normalize(silent)
    assert peak_s == 0.0
    np.testing.assert_array_equal(out_s, silent)


def test_load_audio_keeps_native_sr(tmp_path):
    for sr_in in (44100, 48000):
        t = np.arange(sr_in, dtype=np.float32) / sr_in
        stereo = np.stack([0.2 * np.sin(2 * np.pi * 440 * t),
                           0.1 * np.sin(2 * np.pi * 880 * t)], axis=1)
        path = tmp_path / f'tone_{sr_in}.wav'
        sf.write(str(path), stereo, sr_in)
        y, sr = load_audio(path, normalize=True)
        assert sr == sr_in
        assert y.ndim == 1 and y.dtype == np.float32
        assert float(np.max(np.abs(y))) == pytest.approx(PEAK_TARGET, rel=1e-3, abs=1e-3)


def test_load_audio_missing_file(tmp_path):
    with pytest.raises(AudioLoadError):
        load_audio(tmp_path / 'nope.wav')


def test_qc_flags_hard_and_soft():
    short = waveform_stats(np.ones(int(TEST_SR * 1.0), dtype=np.float32), sr=TEST_SR)
    hard, soft = qc_flags(short, meta_duration=1.0)
    assert 'too_short' in hard

    silent = waveform_stats(np.zeros(int(TEST_SR * 5), dtype=np.float32), sr=TEST_SR)
    hard, soft = qc_flags(silent, meta_duration=5.0)
    assert 'silence' in hard

    y = 0.5 * np.ones(int(TEST_SR * 5), dtype=np.float32)
    stats = waveform_stats(y, sr=TEST_SR)
    hard, soft = qc_flags(stats, meta_duration=None)
    assert hard == []
    assert 'missing_meta_duration' in soft

    hard, soft = qc_flags(stats, meta_duration=10.0)
    assert 'duration_mismatch' in soft

    long_y = np.full(int(TEST_SR * 601), 0.1, dtype=np.float32)
    hard, soft = qc_flags(waveform_stats(long_y, sr=TEST_SR), meta_duration=601.0)
    assert hard == []
    assert 'long_duration' in soft


def test_duration_rel_err():
    assert duration_rel_err(110, 100) == pytest.approx(0.1)
    assert duration_rel_err(10, None) is None


def test_inventory_alignment(tmp_path):
    audio = tmp_path / 'mp3_pieces'
    audio.mkdir()
    (audio / 'a.Piece One.mp3').write_bytes(b'x')
    (audio / 'b.Piece Two.mp3').write_bytes(b'x')
    (audio / 'orphan.mp3').write_bytes(b'x')
    meta = tmp_path / 'meta.json'
    meta.write_text(json.dumps({
        'a.Piece One': {'ps': '1', 'duration': 12.0, 'composer': 'A'},
        'b.Piece Two': {'ps': '2', 'duration': 20.0, 'composer': 'B'},
        'c.Missing': {'ps': '3', 'duration': 30.0, 'composer': 'C'},
    }), encoding='utf-8')
    stems = {p.stem for p in audio.glob('*.mp3')}
    keys = set(json.loads(meta.read_text(encoding='utf-8')))
    assert len(stems & keys) == 2
    assert sorted(keys - stems) == ['c.Missing']
    assert sorted(stems - keys) == ['orphan']


def test_qc_one_and_assemble(tmp_path, monkeypatch):
    monkeypatch.setattr('preprocess_audio.ROOT', tmp_path)
    audio_dir = tmp_path / 'datasets' / 'audio' / 'mp3_pieces'
    audio_dir.mkdir(parents=True)
    y = 0.25 * np.sin(2 * np.pi * 440 * np.arange(TEST_SR * 4) / TEST_SR).astype(np.float32)
    wav_path = audio_dir / 'demo.wav'
    _write_wav(wav_path, y)
    row = qc_one('demo', wav_path, meta_duration=4.0)
    assert row['ok'] is True

    silent_path = audio_dir / 'silent.wav'
    _write_wav(silent_path, np.zeros(TEST_SR * 4, dtype=np.float32))
    bad = qc_one('silent', silent_path, meta_duration=4.0)
    assert bad['ok'] is False and 'silence' in bad['flags']

    split = {'seed': 1, 'parts': {'train': ['demo'], 'val': ['silent'], 'test': ['x']}}
    split_path = tmp_path / 'split.json'
    split_path.write_text(json.dumps(split), encoding='utf-8')
    inventory = {
        'audio_dir': 'datasets/audio/mp3_pieces',
        'metadata_path': 'meta.json',
        'n_metadata': 2,
        'n_audio_files': 2,
        'exact_match': 2,
        'missing_audio': 0,
        'orphan_audio': 0,
        'missing_audio_keys': [],
        'orphan_audio_keys': [],
    }
    report = assemble_report(inventory, {'demo': row, 'silent': bad},
                             ['demo', 'silent'])
    assert report['n_ok'] == 1 and report['n_failed'] == 1
    assert report['ok'][0]['key'] == 'demo'
    assert report['failed'][0]['key'] == 'silent'
    subset = build_common_subset([r['key'] for r in report['ok']], split_path)
    assert subset['parts']['train'] == ['demo']
    assert subset['parts']['val'] == []


def test_run_qc_resume(tmp_path, monkeypatch):
    monkeypatch.setattr('preprocess_audio.ROOT', tmp_path)
    audio_rel = Path('datasets/audio/mp3_pieces')
    audio_dir = tmp_path / audio_rel
    audio_dir.mkdir(parents=True)
    for name, amp in [('a', 0.3), ('b', 0.4)]:
        y = amp * np.sin(2 * np.pi * 220 * np.arange(TEST_SR * 4) / TEST_SR).astype(np.float32)
        _write_wav(audio_dir / f'{name}.wav', y)

    inventory = {
        'audio_dir': str(audio_rel).replace('\\', '/'),
        'metadata_path': 'meta.json',
        'n_metadata': 2,
        'n_audio_files': 2,
        'exact_match': 2,
        'missing_audio': 0,
        'orphan_audio': 0,
        'missing_audio_keys': [],
        'orphan_audio_keys': [],
        'matched': [
            {'key': 'a', 'path': str(audio_rel / 'a.wav').replace('\\', '/'), 'bytes': 1,
             'meta_duration': 4.0, 'ps': '1'},
            {'key': 'b', 'path': str(audio_rel / 'b.wav').replace('\\', '/'), 'bytes': 1,
             'meta_duration': 4.0, 'ps': '2'},
        ],
    }
    split = {'seed': 1, 'parts': {'train': ['a'], 'val': ['b'], 'test': []}}
    # empty test not allowed by build_common_subset validate... it allows empty lists
    # actually validate says if not keys - ValueError for empty. Fix split.
    split = {'seed': 1, 'parts': {'train': ['a', 'z'], 'val': ['b'], 'test': ['y']}}
    split_path = tmp_path / 'split.json'
    split_path.write_text(json.dumps(split), encoding='utf-8')

    report = tmp_path / 'qc.json'
    first = run_qc(inventory, report, limit=1, resume=False)
    assert first['n_queued'] == 1 and [r['key'] for r in first['ok']] == ['a']

    second = run_qc(inventory, report, resume=True)
    assert second['n_queued'] == 2
    assert [r['key'] for r in second['ok']] == ['a', 'b']
    assert second['n_failed'] == 0
    subset = build_common_subset([r['key'] for r in second['ok']], split_path)
    assert subset['counts']['train'] == 1 and subset['counts']['val'] == 1


def test_common_subset_intersection(tmp_path):
    split = {
        'seed': 42,
        'parts': {
            'train': ['a', 'b', 'c'],
            'val': ['d'],
            'test': ['e', 'f'],
        },
    }
    split_path = tmp_path / 'split.json'
    split_path.write_text(json.dumps(split), encoding='utf-8')
    valid = ['a', 'd', 'e', 'z']
    payload = build_common_subset(valid, split_path)
    assert payload['counts'] == {'train': 1, 'val': 1, 'test': 1}
    assert payload['source_counts'] == {'train': 3, 'val': 1, 'test': 2}
    assert payload['missing_counts'] == {'train': 2, 'val': 0, 'test': 1}
    assert payload['n_missing'] == 3
    assert payload['parts'] == {'train': ['a'], 'val': ['d'], 'test': ['e']}
    assert payload['missing_vs_source'] == {'train': ['b', 'c'], 'val': [], 'test': ['f']}
    assert payload['n_total'] == 3
