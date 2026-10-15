"""Tests for native-resolution two-branch CRNN late fusion."""
import hashlib
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from data.native_sequences import (NativePairSource,NativeTwoBranchDataset,collate_native_pairs,
    compute_native_statistics,load_native_normalization,rhythm_validity)
from features_cqt import sha256
from models.model_late_fusion import LateFusionCRNN
from test_crnn_rhythm import _fixture


def test_rhythm_validity_rules():
    x=np.ones((9,4));x[4,1]=0;x[0,2]=0;x[1,3]=0
    v=rhythm_validity(x)
    assert not v[4:7,1].any() and not v[7,1]
    assert not v[7:,2].any() and not v[7:,3].any()
    assert v[:4].all()


def test_native_dataset_preserves_independent_lengths(tmp_path):
    meta,split,manifest,cqt_dir,rhythm_dir,ids,labels=_fixture(tmp_path,('a','b','c'))
    source=NativePairSource(ids,labels,cqt_dir,rhythm_dir,meta,split,manifest)
    cm,cs,cc,rm,rs,rc,cf,rw=compute_native_statistics(source,['a'])
    ds=NativeTwoBranchDataset(ids,labels,source,cm,cs,rm,rs)
    c,cl,r,rl,y,keys=collate_native_pairs([ds[0],ds[1]])
    assert cl.tolist()==[40,41]
    assert rl.tolist()==[41,41]
    assert c.shape==(2,88,41) and r.shape==(2,9,41)
    assert torch.isfinite(c).all() and torch.isfinite(r).all()


def test_late_fusion_has_independent_branches_no_scalar_feature_weight():
    model=LateFusionCRNN(mode='late-fusion',cqt_conv_channels=4,cqt_hidden_size=3,
                         rhythm_conv_channels=5,rhythm_hidden_size=2,embedding_size=6,dropout=0)
    names=dict(model.named_parameters())
    assert any(k.startswith('cqt_branch.') for k in names)
    assert any(k.startswith('rhythm_branch.') for k in names)
    assert not any('weight_gate' in k or 'alpha' in k for k in names)
    c=torch.randn(2,88,25);r=torch.randn(2,9,13)
    score,representation,details=model(c,torch.tensor([20,25]),r,torch.tensor([9,13]),True)
    assert score.shape==(2,) and representation.shape==(2,12)
    assert set(details)=={'cqt','rhythm'}
    score.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize('mode,expected', [('cqt-only',{'cqt'}),('rhythm-only',{'rhythm'}),
                                           ('late-fusion',{'cqt','rhythm'})])
def test_ablation_modes_only_build_expected_branches(mode,expected):
    model=LateFusionCRNN(mode=mode,cqt_conv_channels=4,cqt_hidden_size=3,
                         rhythm_conv_channels=4,rhythm_hidden_size=3,embedding_size=5,dropout=0)
    present=set()
    if model.cqt_branch:present.add('cqt')
    if model.rhythm_branch:present.add('rhythm')
    assert present==expected
    out=model(torch.randn(2,88,20),torch.tensor([18,20]),
              torch.randn(2,9,14),torch.tensor([12,14]))
    assert out.shape==(2,)


def test_train_only_statistics_and_normalization_provenance(tmp_path):
    meta,split,manifest,cqt_dir,rhythm_dir,ids,labels=_fixture(tmp_path,('a','b','c'))
    source=NativePairSource(ids,labels,cqt_dir,rhythm_dir,meta,split,manifest)
    cm,cs,cc,rm,rs,rc,cf,rw=compute_native_statistics(source,['a'])
    norm=tmp_path/'norm.npz'
    np.savez(norm,cqt_mean=cm,cqt_std=cs,cqt_counts=cc,rhythm_mean=rm,rhythm_std=rs,
        rhythm_counts=rc,mode=np.asarray('native_two_branch_v1'),source_signature=np.asarray(source.source_signature),
        split_sha256=np.asarray(sha256(split)),metadata_sha256=np.asarray(sha256(meta)),
        train_ids_sha256=np.asarray(hashlib.sha256(b'a').hexdigest()))
    got=load_native_normalization(norm,source,split,meta,['a'])
    for a,b in zip(got,(cm,cs,rm,rs)):np.testing.assert_allclose(a,b)
    with pytest.raises(ValueError,match='provenance'):
        load_native_normalization(norm,source,split,meta,['b'])


def test_prepare_and_three_cli_modes(tmp_path):
    meta,split,manifest,cqt_dir,rhythm_dir,ids,labels=_fixture(tmp_path,('a','b','c'))
    args=['--metadata',str(meta),'--split',str(split),'--manifest',str(manifest),
          '--cqt-dir',str(cqt_dir),'--rhythm-dir',str(rhythm_dir)]
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    def run(script,extra):
        result=subprocess.run([sys.executable,'-B','-W','error',str(ROOT/'src'/script),*extra],
            capture_output=True,text=True,env=env,timeout=120)
        assert result.returncode==0,result.stdout+'\n'+result.stderr
    norm=tmp_path/'native_norm.npz';run('prepare_late_fusion.py',args+['--output',str(norm)])
    with np.load(norm,allow_pickle=False) as d:
        assert d['cqt_mean'].shape==(88,) and d['rhythm_mean'].shape==(9,)
        assert int(d['cqt_frames'])==40 and int(d['rhythm_windows'])==41
    for mode in ('cqt-only','rhythm-only','late-fusion'):
        output=tmp_path/('run_'+mode)
        run('train_late_fusion.py',args+['--normalization',str(norm),'--mode',mode,
            '--device','cpu','--num-workers','0','--batch-size','2','--max-epochs','1','--patience','1',
            '--cqt-conv-channels','4','--cqt-hidden-size','3','--rhythm-conv-channels','4',
            '--rhythm-hidden-size','3','--embedding-size','5','--dropout','0','--no-amp',
            '--output',str(output)])
        config=json.loads((output/'config.json').read_text())
        assert config['architecture']['fixed_feature_weights'] is False
        assert config['native_time_axes'] is True and config['rhythm_interpolated_to_cqt'] is False
        assert config['feature_group'].endswith(mode.replace('-','_'))
        assert json.loads((output/'metrics.json').read_text())['test_evaluated'] is False
        artifact=torch.load(output/'model.pt',map_location='cpu',weights_only=True)
        assert artifact['config']['architecture']['fusion']=='representation concatenation'
