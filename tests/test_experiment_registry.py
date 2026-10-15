"""Tests for non-destructive experiment registry generation."""
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from summarize_experiments import build


def _hashes(path):
    return {str(f.relative_to(path)):hashlib.sha256(f.read_bytes()).hexdigest()
            for f in path.rglob('*') if f.is_file()}


def test_registry_does_not_change_existing_run_artifacts(tmp_path):
    outputs=tmp_path/'outputs';run=outputs/'family'/'run_01';run.mkdir(parents=True)
    (run/'metrics.json').write_text(json.dumps({'validation':{'n':2,'kendall_tau_c':.5,
        'spearman':.6,'mae':1.,'mse':2.,'acc_pm1':.5},'test_evaluated':False}))
    (run/'config.json').write_text(json.dumps({'model':'m','feature_group':'f','split_sha256':'s',
        'training':{'seed':42,'best_epoch':3},'runtime_seconds':1.2}))
    (run/'predictions_val.json').write_text('[]')
    before=_hashes(run);payload=build(outputs);after=_hashes(run)
    assert before==after and payload['n_runs']==1
    assert (outputs/'README.md').exists()
    assert (outputs/'summaries/experiment_index.json').exists()
    assert (outputs/'summaries/experiment_index.csv').exists()
    row=json.loads((outputs/'summaries/experiment_index.json').read_text())['runs'][0]
    assert row['split_sha256']=='s' and row['n_validation']==2 and row['status']=='complete'
