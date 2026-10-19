"""The v2 suite adds one control without changing the original nine experiments."""
import copy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from data.aligned_sequences import AlignedSequenceDataset
from experiments import suite
from experiments.evaluate_test import evaluate_release
from test_final_test_evaluation import fixture

V2 = ROOT/'configs/rerun_v2.json'


def test_v2_adds_only_early_rhythm_and_preserves_all_v1_parameters():
    old, new = suite.read_json(suite.CATALOGUE), suite.read_json(V2)
    reduced = copy.deepcopy(new)
    reduced['suite'] = old['suite']
    added = [r for r in reduced['experiments'] if r['id'] == 'early_rhythm']
    reduced['experiments'] = [r for r in reduced['experiments'] if r['id'] != 'early_rhythm']
    assert reduced == old
    assert len(added) == 1 and added[0]['args'] == ['--feature-mode', 'rhythm']
    plans = [suite.plan(new, row) for row in new['experiments']]
    assert len(plans) == len({p['run'] for p in plans}) == 10
    assert [len(suite.select(new, g)) for g in ('hgb', 'early', 'late')] == [4, 3, 3]
    for p in plans:
        assert p['run'].is_relative_to(ROOT/'outputs/rerun_v2')
        assert '--evaluate-test' not in p['command']


def test_rhythm_only_is_exact_fused_tail_and_cannot_see_cqt_values():
    class Source:
        def __init__(self):
            self.values = np.arange(97*13, dtype=np.float32).reshape(97, 13)
            self.valid = np.ones((97, 13), dtype=bool)
            self.valid[92:96, 3:7] = False
        def read(self, key):
            return self.values.copy(), self.valid.copy()
    source = Source()
    mean, std = np.arange(97), np.arange(1, 98)
    args = (['one'], [3], source, mean, std)
    both = AlignedSequenceDataset(*args, True)
    cqt = AlignedSequenceDataset(*args, False)
    rhythm = AlignedSequenceDataset(*args, feature_mode='rhythm')
    assert both.lengths == cqt.lengths == rhythm.lengths == [13]
    x = both[0][0]
    torch.testing.assert_close(cqt[0][0], x[:88], rtol=0, atol=0)
    torch.testing.assert_close(rhythm[0][0], x[88:], rtol=0, atol=0)
    assert rhythm.channels == 9 and not rhythm[0][0][4:8, 3:7].any()
    source.values[:88] += 100000
    torch.testing.assert_close(rhythm[0][0], x[88:], rtol=0, atol=0)
    with pytest.raises(ValueError, match='Unknown'):
        AlignedSequenceDataset(*args, feature_mode='typo')
    with pytest.raises(ValueError, match='only one'):
        AlignedSequenceDataset(*args, True, feature_mode='rhythm')


def test_v2_cli_dry_run_and_test_output_are_isolated():
    result = subprocess.run([sys.executable, str(ROOT/'scripts/run_experiments.py'),
                             '--config', str(V2), '--dry-run'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.count(' --output ') == 10
    assert 'early_rhythm' in result.stdout and '/outputs/rerun_v1/' not in result.stdout
    result = subprocess.run([sys.executable, str(ROOT/'scripts/evaluate_test.py'),
                             '--config', str(V2)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'all 10 frozen models' in result.stdout
    assert 'outputs/rerun_v2/test_evaluation/final_01' in result.stdout


def test_v2_all_ten_train_and_test_cli_paths_on_synthetic_data(tmp_path, monkeypatch):
    # Actual trainers, reduced budgets, synthetic partitions only (44/11/11 pieces).
    cat = fixture(tmp_path, monkeypatch, catalogue_path=V2)
    summary = suite.summarize(cat, root=tmp_path)
    assert len(summary['runs']) == 10
    assert all(row['status'] == 'completed' for row in summary['runs'])
    assert not (tmp_path/'outputs/rerun_v1').exists()
    for row in cat['experiments']:
        item = suite.plan(cat, row, root=tmp_path)
        state = suite.read_json(item['run']/'run.json')
        assert state['catalogue_sha256'] == suite.sha(V2)
        config = suite.read_json(item['run']/'artifacts/config.json')
        metrics = suite.read_json(item['run']/'artifacts/metrics.json')
        assert metrics['test_evaluated'] is False and 'test' not in metrics
        assert not (item['run']/'artifacts/predictions_test.json').exists()
        if row['id'] == 'early_rhythm':
            assert config['feature_mode'] == 'rhythm'
            assert config['feature_group'] == 'rhythm_sequence_aligned'
            assert config['architecture']['input_bins'] == 9
    result = evaluate_release(cat, data_root=tmp_path, code_root=ROOT, device_name='cpu',
                              workers=0, batch_size=2, catalogue_path=V2)
    assert result['release']['status'] == 'completed'
    assert result['release']['catalogue_sha256'] == suite.sha(V2)
    assert result['release']['training_performed'] is False
    assert len(result['runs']) == 10
    assert all(row['test']['n'] == 11 for row in result['runs'])
    assert (tmp_path/'outputs/rerun_v2/test_evaluation/final_01/early/rhythm_only/predictions_test.json').is_file()
    with pytest.raises(FileExistsError, match='already completed'):
        evaluate_release(cat, data_root=tmp_path, code_root=ROOT, device_name='cpu',
                         workers=0, batch_size=2, catalogue_path=V2)
