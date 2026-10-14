"""Early stopping tests: deterministic metric sequences and real tiny CPU training."""
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import train_crnn
from features_cqt import sha256
from test_crnn_rhythm import _fixture


def test_plateau_stops_after_exact_patience():
    stop = train_crnn.EarlyStopping(patience=15, min_delta=.001)
    assert not stop.step(.5)
    for _ in range(14):
        assert not stop.step(.5)
    assert stop.step(.5)
    assert stop.stale_epochs == 15


def test_cumulative_small_gains_reset_only_after_threshold():
    stop = train_crnn.EarlyStopping(patience=3, min_delta=.001)
    assert not stop.step(.5)
    assert not stop.step(.5004) and stop.stale_epochs == 1
    assert not stop.step(.5008) and stop.stale_epochs == 2
    assert not stop.step(.5012) and stop.stale_epochs == 0
    assert stop.reference_tau == .5012
    assert not stop.step(.4)
    assert not stop.step(.5013)
    assert stop.step(.5014)


def test_zero_delta_still_requires_strict_improvement():
    stop = train_crnn.EarlyStopping(patience=2, min_delta=0)
    assert not stop.step(.4)
    assert not stop.step(.40000001)
    assert not stop.step(.40000001)
    assert stop.step(.40000001)


def test_undefined_tau_does_not_reset_patience():
    stop = train_crnn.EarlyStopping(patience=2)
    assert not stop.step(None)
    assert stop.step(None)


@pytest.mark.parametrize('patience,delta', [(0,.001),(-1,.001),(3,-.1),(3,float('nan')),(3,float('inf'))])
def test_invalid_stop_parameters(patience,delta):
    with pytest.raises(ValueError):
        train_crnn.EarlyStopping(patience,delta)


def test_nonfinite_tau_fails_loudly():
    with pytest.raises(ValueError):
        train_crnn.EarlyStopping().step(float('nan'))


@pytest.mark.parametrize('max_epochs,patience,expected_epochs,expected_reason,expected_best', [
    (100,3,4,'early_stopping',3),
    (2,15,2,'max_epochs',2),
])
def test_training_stops_restores_true_best_and_persists_logs(
        tmp_path,monkeypatch,max_epochs,patience,expected_epochs,expected_reason,expected_best):
    meta,split,manifest,cqt_dir,rhythm_dir,ids,labels = _fixture(tmp_path,('a','b','c'))
    norm = tmp_path/'norm.npz'
    np.savez(norm,mean=np.zeros(88),std=np.ones(88),
             split_sha256=np.asarray(sha256(split)),metadata_sha256=np.asarray(sha256(meta)))
    args = Namespace(output=tmp_path/'run',batch_size=2,max_epochs=max_epochs,patience=patience,
        hidden_size=3,conv_channels=4,num_workers=0,prefetch_factor=2,
        learning_rate=.001,weight_decay=.0001,grad_clip=1.,dropout=0.,
        min_delta=.001,seed=42,metadata=meta,split=split,rhythm_dir=None,
        feature_mode='cqt',manifest=None,cqt_fps=5.,cqt_time_offset=0.,evaluate_test=False,
        normalization=norm,cqt_dir=cqt_dir,device='cpu',pin_memory=False,amp=False)
    snapshots=[]
    def predict(model,loader,device,amp_enabled=False):
        snapshots.append({k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
        return ['b'],np.array([3.]),np.array([3.])
    values=iter([(.5,4.),(.5002,3.),(.5004,2.),(.5004,2.)])
    def metrics(*args):
        tau,mse=next(values)
        return {'n':1,'kendall_tau_c':tau,'spearman':tau,'mae':1.,'mse':mse,
                'acc_pm1':1.,'correlation_note':None}
    monkeypatch.setattr(train_crnn,'predict',predict)
    monkeypatch.setattr(train_crnn,'evaluate',metrics)
    train_crnn.main(args)
    config=json.loads((args.output/'config.json').read_text())
    assert config['training']['epochs_run']==expected_epochs
    assert config['training']['stop_reason']==expected_reason
    assert config['training']['best_epoch']==expected_best
    assert config['training']['max_epochs']==max_epochs
    history=json.loads((args.output/'history.json').read_text())
    stream=[json.loads(line) for line in (args.output/'history.jsonl').read_text().splitlines()]
    assert stream==history and len(history)==expected_epochs
    checkpoint=torch.load(args.output/'best_checkpoint.pt',map_location='cpu',weights_only=True)
    final=torch.load(args.output/'model.pt',map_location='cpu',weights_only=True)
    assert checkpoint['epoch']==expected_best
    for key,value in snapshots[expected_best-1].items():
        torch.testing.assert_close(value,final['state_dict'][key],rtol=0,atol=0)
        torch.testing.assert_close(value,checkpoint['state_dict'][key],rtol=0,atol=0)
        torch.testing.assert_close(value,snapshots[-1][key],rtol=0,atol=0)
    assert not (args.output/'predictions_test.json').exists()
