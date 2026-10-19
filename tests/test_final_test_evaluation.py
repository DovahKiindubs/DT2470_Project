"""Synthetic regression tests for the one-shot frozen test evaluator."""
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
from data.aligned_sequences import RHYTHM_NAMES
from data.prepare_crnn import main as prepare_early
from data.prepare_late_fusion import main as prepare_late
from experiments import suite
from experiments.evaluate_test import ACKNOWLEDGEMENT,evaluate_release
from argparse import Namespace


def fixture(tmp_path,monkeypatch,catalogue_path=None):
    cat=copy.deepcopy(suite.read_json(catalogue_path or suite.CATALOGUE))
    ids=[f'composer{c}_grade{g}' for c in range(6) for g in range(11)];labels=np.tile(np.arange(11),6)
    meta={k:{'ps':str(int(y)),'composer':k.split('_')[0],'youtube_link':''} for k,y in zip(ids,labels)}
    mp=tmp_path/cat['metadata'];mp.parent.mkdir(parents=True);suite.write_json(mp,meta)
    sp=tmp_path/cat['split'];sp.parent.mkdir(parents=True);parts={'train':ids[:44],'val':ids[44:55],'test':ids[55:]};cat['counts']={k:len(v) for k,v in parts.items()};suite.write_json(sp,{'seed':42,'metadata_sha256':suite.sha(mp),'parts':parts})
    fr=tmp_path/'datasets/features';fr.mkdir()
    for name,dim in [('cqt_hard_windows.npz',528),('rhythm_stats.npz',410),('structure_stats.npz',73)]:
        x=np.random.default_rng(dim).normal(size=(66,dim));x[:,0]=labels
        np.savez(fr/name,X=x,y=labels,ids=np.asarray(ids),feature_names=np.asarray([f'{name}_{i}' for i in range(dim)]),metadata_sha256=np.asarray(suite.sha(mp)))
    cqt=fr/'cqt';rhythm=fr/'rhythm_windows';cqt.mkdir();rhythm.mkdir();records=[]
    for i,(key,y) in enumerate(zip(ids,labels)):
        a=np.random.default_rng(i).normal(size=(88,40+i%5)).astype(np.float32)+float(y);(cqt/(key+'.bin')).write_bytes(pickle.dumps(a,protocol=4))
        x=np.ones((9,20))*(1+y/10);np.savez(rhythm/(key+'.npz'),X=x,times=np.arange(20)*.2,y=np.asarray(y),feature_names=np.asarray(RHYTHM_NAMES),win_sec=np.asarray(4.),hop_sec=np.asarray(.2),audio_sha256=np.asarray('a'*64))
        records.append({'id':key,'label':int(y),'status':'included','cqt_sha256':suite.sha(cqt/(key+'.bin')),'rhythm_sha256':suite.sha(rhythm/(key+'.npz'))})
    np.savez(rhythm/'index.npz',ids=np.asarray(ids),y=labels,lengths=np.full(66,20),feature_names=np.asarray(RHYTHM_NAMES),win_sec=np.asarray(4.),hop_sec=np.asarray(.2),audio_sha256=np.asarray(['a'*64]*66),metadata_sha256=np.asarray(suite.sha(mp)))
    manifest=tmp_path/cat['manifest'];suite.write_json(manifest,{'split_sha256':suite.sha(sp),'records':records,'source_sha256':{'datasets/features/rhythm_windows/index.npz':suite.sha(rhythm/'index.npz')}})
    args=Namespace(metadata=mp,split=sp,manifest=manifest,cqt_dir=cqt,rhythm_dir=rhythm,cqt_fps=5.,cqt_time_offset=0.,output=fr/'cqt_rhythm_common_v1_normalization.npz');prepare_early(args);args.output=fr/'cqt_rhythm_native_normalization.npz';prepare_late(args)
    for key in cat['inputs']:cat['inputs'][key]=suite.sha(tmp_path/key)
    monkeypatch.setattr(suite,'ROOT',ROOT)
    pre=suite.preflight(cat,cat['experiments'],root=tmp_path)
    for row in cat['experiments']:
        item=suite.plan(cat,row,device='cpu',workers=0,root=tmp_path);cmd=item['command']
        for i,value in enumerate(cmd):
            if value.startswith('datasets/'):cmd[i]=str(tmp_path/value)
        if row['group']=='hgb':cmd[cmd.index('--max-iter')+1]='3'
        else:
            cmd[cmd.index('--max-epochs')+1]='1';cmd+=['--no-amp']
            for flag in ('--conv-channels','--hidden-size','--cqt-conv-channels','--cqt-hidden-size','--rhythm-conv-channels','--rhythm-hidden-size','--embedding-size'):
                if flag in cmd:cmd[cmd.index(flag)+1]='4'
        absolute=copy.deepcopy(cat);absolute['metadata']=str(mp);absolute['split']=str(sp);absolute['inputs'][absolute['split']]=cat['inputs'][cat['split']]
        suite.launch(item,absolute,42,pre,threads=1,catalogue_path=catalogue_path)
    return cat


def test_cli_requires_explicit_acknowledgement():
    cmd=[sys.executable,str(ROOT/'scripts/evaluate_test.py'),'--execute']
    result=subprocess.run(cmd,capture_output=True,text=True)
    assert result.returncode!=0 and ACKNOWLEDGEMENT in result.stderr
    result=subprocess.run([sys.executable,str(ROOT/'scripts/evaluate_test.py')],capture_output=True,text=True)
    assert result.returncode==0 and 'does not train' in result.stdout


def test_all_nine_saved_models_are_tested_once(tmp_path,monkeypatch):
    cat=fixture(tmp_path,monkeypatch)
    import experiments.evaluate_test as target
    monkeypatch.setattr(target,'ROOT',ROOT);monkeypatch.setattr(target,'CATALOGUE',suite.CATALOGUE)
    output=tmp_path/'final_test'
    result=evaluate_release(cat,data_root=tmp_path,code_root=ROOT,output=output,device_name='cpu',workers=0,batch_size=2)
    assert result['release']['status']=='completed'
    assert result['release']['training_performed'] is False and result['release']['model_selection_performed'] is False
    assert len(result['runs'])==9 and all(r['status']=='completed' and r['test']['n']==11 for r in result['runs'])
    expected=set(json.loads((tmp_path/cat['split']).read_text())['parts']['test'])
    for row in cat['experiments']:
        rows=json.loads((output/row['group']/row['folder']/'predictions_test.json').read_text())
        assert {r['id'] for r in rows}==expected
    with pytest.raises(FileExistsError,match='already completed'):
        evaluate_release(cat,data_root=tmp_path,code_root=ROOT,output=output,device_name='cpu',workers=0,batch_size=2)


def test_resume_rejects_changed_model_or_data(tmp_path,monkeypatch):
    cat=fixture(tmp_path,monkeypatch)
    import experiments.evaluate_test as target
    monkeypatch.setattr(target,'ROOT',ROOT);monkeypatch.setattr(target,'CATALOGUE',suite.CATALOGUE)
    output=tmp_path/'partial';output.mkdir();suite.write_json(output/'release.json',{'status':'failed','signature':{'bad':True}})
    with pytest.raises(ValueError,match='changed since partial test'):
        evaluate_release(cat,data_root=tmp_path,code_root=ROOT,output=output,device_name='cpu',workers=0,batch_size=2,resume=True)
