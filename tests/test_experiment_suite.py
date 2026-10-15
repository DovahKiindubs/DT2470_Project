"""Safe runner tests; all fits use tiny synthetic data in pytest temporary directories."""
import copy
import json
import os
import pickle
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from experiments import suite
from reporting.metrics import evaluate
from data.aligned_sequences import RHYTHM_NAMES
from data.prepare_crnn import main as prepare_early
from data.prepare_late_fusion import main as prepare_late
from argparse import Namespace


def test_nine_plans_are_unique_and_same_split(tmp_path):
    cat=suite.read_json(suite.CATALOGUE)
    plans=[suite.plan(cat,row,root=tmp_path) for row in cat['experiments']]
    assert len(plans)==len({p['run'] for p in plans})==9
    assert [len(suite.select(cat,g)) for g in ('hgb','early','late')]==[4,2,3]
    for item in plans:
        cmd=item['command']
        assert cmd[cmd.index('--split')+1]==cat['split']
        assert '--evaluate-test' not in cmd
        assert not item['run'].exists()
    early=[p['command'] for p in plans if p['group']=='early']
    assert all('--rhythm-dir' in cmd for cmd in early)
    for name in ('../oops','a/b','a b',''):
        with pytest.raises(ValueError):suite.plan(cat,cat['experiments'][0],run_name=name)
    with pytest.raises(ValueError):suite.select(cat,experiments=['typo'])


def test_default_dry_run_and_unknown_argument():
    result=subprocess.run([sys.executable,str(ROOT/'scripts/run_experiments.py')],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert result.stdout.count(' --output ')==9
    result=subprocess.run([sys.executable,str(ROOT/'scripts/run_experiments.py'),'--evaluate-test'],capture_output=True,text=True)
    assert result.returncode!=0


def test_failure_is_recorded_without_overwriting(tmp_path):
    cat=suite.read_json(suite.CATALOGUE)
    item={'experiment_id':'bad','group':'hgb','run':tmp_path/'failed',
          'command':[sys.executable,'-c','raise RuntimeError("synthetic failure")']}
    with pytest.raises(RuntimeError):suite.launch(item,cat,42,{})
    state=suite.read_json(item['run']/'run.json')
    assert state['status']=='failed'
    assert 'synthetic failure' in (item['run']/'train.log').read_text()
    before=(item['run']/'run.json').read_bytes()
    with pytest.raises(FileExistsError):suite.launch(item,cat,42,{})
    assert (item['run']/'run.json').read_bytes()==before


def test_summary_does_not_scan_archive_or_overwrite_navigation(tmp_path):
    cat=suite.read_json(suite.CATALOGUE)
    nav=tmp_path/'outputs/README.md';nav.parent.mkdir();nav.write_text('keep navigation')
    old=tmp_path/'archives/old/metrics.json';old.parent.mkdir(parents=True);old.write_text('{"not":"a run"}')
    payload=suite.summarize(cat,root=tmp_path)
    assert len(payload['runs'])==9 and all(r['status']=='not_started' for r in payload['runs'])
    assert nav.read_text()=='keep navigation'
    assert old.read_text()=='{"not":"a run"}'


def synthetic_suite(tmp_path):
    cat=copy.deepcopy(suite.read_json(suite.CATALOGUE))
    ids=[f'composer{c}_grade{g}' for c in range(6) for g in range(11)]
    labels=np.tile(np.arange(11),6)
    metadata={k:{'ps':str(int(y)),'composer':k.split('_')[0],'youtube_link':''} for k,y in zip(ids,labels)}
    meta=tmp_path/cat['metadata'];meta.parent.mkdir(parents=True);suite.write_json(meta,metadata)
    split_path=tmp_path/cat['split'];split_path.parent.mkdir(parents=True)
    parts={'train':ids[:44],'val':ids[44:55],'test':ids[55:]}
    cat['counts']={k:len(v) for k,v in parts.items()}
    suite.write_json(split_path,{'seed':42,'metadata_sha256':suite.sha(meta),'parts':parts})
    feature_root=tmp_path/'datasets/features';feature_root.mkdir()
    for filename,dim in [('cqt_hard_windows.npz',528),('rhythm_stats.npz',410),('structure_stats.npz',73)]:
        x=np.random.default_rng(dim).normal(size=(66,dim));x[:,0]=labels
        np.savez(feature_root/filename,X=x,y=labels,ids=np.asarray(ids),
                 feature_names=np.asarray([f'{filename}_{i}' for i in range(dim)]),
                 metadata_sha256=np.asarray(suite.sha(meta)))
    cqt=feature_root/'cqt';rhythm=feature_root/'rhythm_windows';cqt.mkdir();rhythm.mkdir()
    records=[];lengths=[]
    for i,(key,y) in enumerate(zip(ids,labels)):
        array=np.random.default_rng(i).normal(size=(88,40+i%5)).astype(np.float32)+float(y)
        (cqt/(key+'.bin')).write_bytes(pickle.dumps(array,protocol=4))
        times=np.arange(20)*.2;x=np.ones((9,20))*(1+y/10);lengths.append(20)
        np.savez(rhythm/(key+'.npz'),X=x,times=times,y=np.asarray(y),feature_names=np.asarray(RHYTHM_NAMES),
                 win_sec=np.asarray(4.),hop_sec=np.asarray(.2),audio_sha256=np.asarray('a'*64))
        records.append({'id':key,'label':int(y),'status':'included','cqt_sha256':suite.sha(cqt/(key+'.bin')),
                        'rhythm_sha256':suite.sha(rhythm/(key+'.npz'))})
    np.savez(rhythm/'index.npz',ids=np.asarray(ids),y=labels,lengths=np.asarray(lengths),
             feature_names=np.asarray(RHYTHM_NAMES),win_sec=np.asarray(4.),hop_sec=np.asarray(.2),
             audio_sha256=np.asarray(['a'*64]*len(ids)),metadata_sha256=np.asarray(suite.sha(meta)))
    manifest=tmp_path/cat['manifest']
    suite.write_json(manifest,{'split_sha256':suite.sha(split_path),'records':records,
                               'source_sha256':{'datasets/features/rhythm_windows/index.npz':suite.sha(rhythm/'index.npz')}})
    args=Namespace(metadata=meta,split=split_path,manifest=manifest,cqt_dir=cqt,rhythm_dir=rhythm,
                   cqt_fps=5.,cqt_time_offset=0.,output=feature_root/'cqt_rhythm_common_v1_normalization.npz')
    prepare_early(args)
    args.output=feature_root/'cqt_rhythm_native_normalization.npz';prepare_late(args)
    for key in cat['inputs']:cat['inputs'][key]=suite.sha(tmp_path/key)
    return cat


def test_nine_real_cli_modes_on_synthetic_data(tmp_path,monkeypatch):
    cat=synthetic_suite(tmp_path)
    # Same planner/launcher and actual training modules; only tiny test budgets are overridden.
    monkeypatch.setattr(suite,'ROOT',ROOT)
    result=suite.preflight(cat,cat['experiments'],root=tmp_path)
    assert result['paired_files_checked']==66 and result['test_scored'] is False
    for row in cat['experiments']:
        item=suite.plan(cat,row,device='cpu',workers=0,root=tmp_path)
        cmd=item['command']
        for i,value in enumerate(cmd):
            if value.startswith('datasets/'):cmd[i]=str(tmp_path/value)
        if row['group']=='hgb':
            cmd[cmd.index('--max-iter')+1]='3'
        else:
            cmd[cmd.index('--max-epochs')+1]='1'
            cmd.extend(['--no-amp'])
            for flag in ('--conv-channels','--hidden-size','--cqt-conv-channels','--cqt-hidden-size',
                         '--rhythm-conv-channels','--rhythm-hidden-size','--embedding-size'):
                if flag in cmd:cmd[cmd.index(flag)+1]='4'
        absolute=copy.deepcopy(cat)
        absolute['metadata']=str(tmp_path/cat['metadata']);absolute['split']=str(tmp_path/cat['split'])
        absolute['inputs'][absolute['split']]=cat['inputs'][cat['split']]
        suite.launch(item,absolute,42,result,threads=1)
        state=suite.read_json(item['run']/'run.json')
        assert state['status']=='completed'
        metrics=suite.read_json(item['run']/'artifacts/metrics.json')
        assert metrics['validation']['n']==11 and metrics['test_evaluated'] is False
        cfg=suite.read_json(item['run']/'artifacts/config.json')
        if row['group']=='hgb':assert cfg['n_features']==row['n_features']
        if row['group']=='early':assert cfg['architecture']['input_bins']==(88 if row['id']=='early_cqt' else 97)
        if row['group']=='late':assert cfg['feature_group'].endswith(row['args'][-1].replace('-','_'))
    payload=suite.summarize(cat,root=tmp_path)
    assert len(payload['runs'])==9 and all(r['status']=='completed' for r in payload['runs'])
    # Corrupt a saved metric and ensure successful process exit is not enough for completion.
    run=item['run'];path=run/'artifacts/metrics.json';metrics=suite.read_json(path)
    metrics['validation']['mae']+=1;suite.write_json(path,metrics)
    with pytest.raises(ValueError,match='Metric mismatch'):suite.verify_result(run,absolute,42)
    # Changing a frozen cache must fail preflight, not silently run on new data.
    with (tmp_path/'datasets/features/structure_stats.npz').open('ab') as stream:stream.write(b'x')
    with pytest.raises(ValueError,match='Frozen input changed'):
        suite.preflight(cat,[cat['experiments'][2]],root=tmp_path)
