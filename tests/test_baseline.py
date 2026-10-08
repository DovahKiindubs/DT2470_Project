"""不依赖真实数据的测试；CLI 测试使用临时合成数据，不写入正式数据目录。"""
import io
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from features_cqt import ArrayUnpickler, extract, sha256, summarize_cqt
from evaluate import evaluate
from split_data import make_groups, make_split, recording_id, validate_split
from train import build_model
from train_hgb import build_hgb


def test_cqt_summary_and_order():
    cqt = np.tile(np.array([-3., -1., 1., 3.]), (88, 1))
    f = summarize_cqt(cqt)
    assert f.shape == (176,)
    np.testing.assert_allclose(f[:88], 0)
    np.testing.assert_allclose(f[88:], np.sqrt(5))


@pytest.mark.parametrize('cqt', [np.zeros((0, 88)), np.zeros((88, 0)),
    np.zeros((3, 88)), np.zeros((88, 2, 1)), np.full((88, 2), np.nan),
    np.full((88, 2), np.inf), np.ones((88, 2), dtype=complex)])
def test_reject_invalid_cqt(cqt):
    with pytest.raises(ValueError):
        summarize_cqt(cqt)


def test_restricted_pickle():
    original = np.ones((88, 4), dtype=np.float32, order='F')
    restored = ArrayUnpickler(io.BytesIO(pickle.dumps(original, protocol=4))).load()
    np.testing.assert_array_equal(original, restored)
    with pytest.raises(pickle.UnpicklingError):
        ArrayUnpickler(io.BytesIO(pickle.dumps(eval))).load()


def test_extraction_alignment_and_no_overwrite(tmp_path):
    cqt = tmp_path/'cqt'
    cqt.mkdir()
    for name, value in [('z', 2), ('a', 1)]:
        (cqt/f'{name}.bin').write_bytes(pickle.dumps(np.full((88, 2), value, dtype=np.float32), protocol=4))
    metadata = tmp_path/'metadata.json'
    metadata.write_text(json.dumps({'z': {'ps': '2'}, 'a': {'ps': '1'}}))
    output = tmp_path/'features.npz'
    extract(cqt, metadata, output)
    with np.load(output, allow_pickle=False) as d:
        assert d['ids'].tolist() == ['a', 'z']
        assert d['y'].tolist() == [1, 2]
        np.testing.assert_allclose(d['X'][:, 0], [1, 2])
    with pytest.raises(FileExistsError):
        extract(cqt, metadata, output)


def test_metrics_perfect_and_reversed():
    result = evaluate([0, 1, 2], [0, 1, 2])
    assert result['kendall_tau_c'] == pytest.approx(1)
    assert result['spearman'] == pytest.approx(1)
    assert result['mse'] == 0 and result['acc_pm1'] == 1
    assert evaluate([0, 1, 2], [2, 1, 0])['kendall_tau_c'] == pytest.approx(-1)


def test_tau_c_ties_not_tau_b():
    # 五个 concordant pair，四个样本，参考标签三个等级：2*5/(16*2/3)=0.9375。
    assert evaluate([0, 0, 1, 2], [0., 0.1, 1., 2.])['kendall_tau_c'] == pytest.approx(0.9375)


def test_metrics_constant_and_rounding():
    result = evaluate([0, 1, 2], [1., 1., 1.])
    assert result['kendall_tau_c'] is None and result['spearman'] is None
    assert result['mse'] == pytest.approx(2/3)
    result = evaluate([0, 10], [1.5, 20.])
    assert result['acc_pm1'] == 0.5
    assert result['mse'] == pytest.approx((2.25+100)/2)


@pytest.mark.parametrize('y,p', [([], []), ([1], [np.nan]), ([1, 2], [1]), ([-1, 0], [0, 0])])
def test_invalid_metrics(y, p):
    with pytest.raises(ValueError):
        evaluate(y, p)


def test_video_id_and_transitive_groups():
    assert recording_id('https://www.youtube.com/embed/abcdefghijk?start=5') == 'abcdefghijk'
    assert recording_id('https://www.youtube.com/watch?v=abcdefghijk&t=5') == 'abcdefghijk'
    assert recording_id('https://youtu.be/abcdefghijk') == 'abcdefghijk'
    meta = {'a': {'composer': 'A ', 'youtube_link': 'https://youtu.be/abcdefghijk'},
            'b': {'composer': 'a', 'youtube_link': ''},
            'c': {'composer': 'B', 'youtube_link': 'https://youtube.com/embed/abcdefghijk'}}
    assert len(set(make_groups(list(meta), meta))) == 1


def test_split_reproducible_and_leakage_guard():
    ids = np.asarray([f'piece{i}' for i in range(40)])
    groups = np.repeat(np.arange(20), 2)
    parts = make_split(ids, groups)
    assert parts == make_split(ids, groups)
    validate_split(parts, ids, groups)
    parts['val'].append(parts['train'][0])
    with pytest.raises(ValueError):
        validate_split(parts, ids, groups)


def test_scaler_uses_training_data_only():
    X = np.array([[0., 2.], [2., 4.], [4., 6.]])
    model = build_model(1).fit(X, [0, 1, 2])
    model.predict(np.array([[1000., 1000.]]))
    np.testing.assert_allclose(model.named_steps['standardscaler'].mean_, [2., 4.])


def test_train_cli_and_model_roundtrip(tmp_path):
    import joblib
    ids = np.asarray([f'c{c}_g{g}' for c in range(10) for g in range(11)])
    meta = {key: {'ps': str(i % 11), 'composer': f'composer{i//11}', 'youtube_link': ''}
            for i, key in enumerate(ids)}
    metadata = tmp_path/'meta.json'
    metadata.write_text(json.dumps(meta))
    y = np.arange(len(ids)) % 11
    X = np.random.default_rng(0).normal(size=(len(ids), 176))
    X[:, 0] = y
    features = tmp_path/'features.npz'
    np.savez(features, X=X, y=y, ids=ids, feature_names=np.asarray([f'f{i}' for i in range(176)]),
             metadata_sha256=np.asarray(sha256(metadata)))
    output, split = tmp_path/'run', tmp_path/'split.json'
    command = [sys.executable, '-W', 'error', str(ROOT/'src/train.py'), '--features', str(features),
               '--metadata', str(metadata), '--split', str(split), '--output', str(output), '--alphas', '1', '10']
    subprocess.run(command, check=True, capture_output=True, text=True)
    metrics = json.loads((output/'metrics.json').read_text())
    assert metrics['test_evaluated'] is False and 'test' not in metrics
    assert not (output/'predictions_test.json').exists()
    rows = json.loads((output/'predictions_val.json').read_text())
    lookup = {key: i for i, key in enumerate(ids)}
    artifact = joblib.load(output/'model.joblib')
    pred = artifact['pipeline'].predict(X[[lookup[row['id']] for row in rows]])
    np.testing.assert_allclose(pred, [row['score'] for row in rows])
    assert subprocess.run(command, check=False, capture_output=True).returncode != 0
    final = tmp_path/'final'
    test_command = command.copy()
    test_command[test_command.index('--output')+1] = str(final)
    subprocess.run(test_command+['--evaluate-test'], check=True, capture_output=True, text=True)
    assert json.loads((final/'metrics.json').read_text())['test_evaluated'] is True


def test_hgb_builder_and_cli(tmp_path):
    import joblib
    model = build_hgb(7, max_iter=5, learning_rate=0.1, random_state=3)
    assert model.max_leaf_nodes == 7 and model.early_stopping is False
    ids = np.asarray([f'c{c}_g{g}' for c in range(10) for g in range(11)])
    meta = {key: {'ps': str(i % 11), 'composer': f'composer{i//11}', 'youtube_link': ''}
            for i, key in enumerate(ids)}
    metadata = tmp_path/'meta.json'
    metadata.write_text(json.dumps(meta))
    y = np.arange(len(ids)) % 11
    X = np.random.default_rng(1).normal(size=(len(ids), 176))
    X[:, 0] = y
    features = tmp_path/'features.npz'
    np.savez(features, X=X, y=y, ids=ids, feature_names=np.asarray([f'f{i}' for i in range(176)]),
             metadata_sha256=np.asarray(sha256(metadata)))
    # 用同一分组函数冻结一个临时划分，HGB 必须复用它而非创建新划分。
    split = tmp_path/'split.json'
    parts = make_split(ids, make_groups(ids, meta))
    split.write_text(json.dumps({'seed': 42, 'metadata_sha256': sha256(metadata), 'parts': parts}))
    output = tmp_path/'hgb'
    command = [sys.executable, '-W', 'error', str(ROOT/'src/train_hgb.py'),
               '--features', str(features), '--metadata', str(metadata), '--split', str(split),
               '--output', str(output), '--leaf-nodes', '7', '15', '--max-iter', '10']
    subprocess.run(command, check=True, capture_output=True, text=True)
    result = json.loads((output/'metrics.json').read_text())
    assert result['test_evaluated'] is False and 'test' not in result
    assert not (output/'predictions_test.json').exists()
    config = json.loads((output/'config.json').read_text())
    assert config['max_leaf_nodes_candidates'] == [7, 15]
    artifact = joblib.load(output/'model.joblib')
    assert artifact['model'].max_leaf_nodes == config['chosen_max_leaf_nodes']
    rows = json.loads((output/'predictions_val.json').read_text())
    lookup = {key: i for i, key in enumerate(ids)}
    pred = artifact['model'].predict(X[[lookup[row['id']] for row in rows]])
    np.testing.assert_allclose(pred, [row['score'] for row in rows])
