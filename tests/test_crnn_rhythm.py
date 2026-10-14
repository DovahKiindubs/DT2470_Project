"""Tests for the shared CQT/rhythm time alignment and 88/97-channel CRNN modes."""
import hashlib
import io
import json
import pickle
import sys
import subprocess
import os
import unicodedata
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

from data.aligned_sequences import (AlignedFeatureSource, AlignedSequenceDataset,
    RHYTHM_NAMES, align_arrays, compute_aligned_statistics, load_aligned_normalization,
    load_split)
from features_cqt import sha256
from models.model_crnn import CQTCRNN, collate_sequences


def _write_pair(tmp_path, key='piece', length=40):
    cqt_dir, rhythm_dir = tmp_path/'cqt', tmp_path/'rhythm'
    cqt_dir.mkdir(exist_ok=True); rhythm_dir.mkdir(exist_ok=True)
    cqt = np.tile(np.arange(length, dtype=np.float32), (88, 1))
    (cqt_dir/f'{key}.bin').write_bytes(pickle.dumps(cqt, protocol=4))
    times = np.arange(0., 8.1, .2)
    x = np.vstack([np.linspace(i+1, i+2, len(times)) for i in range(9)])
    x[4:7, 5] = 0.; x[0:2, 8] = 0.; x[7, 8] = 120.
    audio_hash = 'a'*64
    np.savez(rhythm_dir/f'{key}.npz', X=x, times=times, y=np.asarray(3),
             feature_names=np.asarray(RHYTHM_NAMES), win_sec=np.asarray(4.),
             hop_sec=np.asarray(.2), audio_sha256=np.asarray(audio_hash))
    return cqt_dir, rhythm_dir, cqt, x, times, audio_hash


def _fixture(tmp_path, keys=('piece',)):
    cqt_dir, rhythm_dir = tmp_path/'cqt', tmp_path/'rhythm'
    rows=[]
    metadata={}
    for i,key in enumerate(keys):
        _,_,cqt,x,times,ah=_write_pair(tmp_path,key,40+i)
        metadata[key]={'ps':'3','composer':f'c{i}','youtube_link':''}
        rows.append((key,cqt,x,times,ah))
    meta=tmp_path/'meta.json'; meta.write_text(json.dumps(metadata),encoding='utf-8')
    parts={'train':list(keys[:1]),'val':list(keys[1:2] or keys[:1]),'test':list(keys[2:3] or keys[:1])}
    # For single-key source-only tests, bypass validate_split; source itself only needs the file list.
    split=tmp_path/'split.json'; split.write_text(json.dumps({'metadata_sha256':sha256(meta),'parts':parts}),encoding='utf-8')
    ids=np.asarray(keys); labels=np.full(len(keys),3)
    np.savez(rhythm_dir/'index.npz',ids=ids,y=labels,lengths=np.asarray([r[2].shape[1] for r in rows]),
        feature_names=np.asarray(RHYTHM_NAMES),audio_sha256=np.asarray([r[4] for r in rows]),
        metadata_sha256=np.asarray(sha256(meta)),win_sec=np.asarray(4.),hop_sec=np.asarray(.2))
    manifest={'split_sha256':sha256(split),'source_sha256':{'datasets/features/rhythm_windows/index.npz':sha256(rhythm_dir/'index.npz')},
              'records':[]}
    for key,cqt,x,times,ah in rows:
        manifest['records'].append({'id':key,'label':3,'status':'included',
            'cqt_sha256':sha256(cqt_dir/f'{key}.bin'),'rhythm_sha256':sha256(rhythm_dir/f'{key}.npz')})
    mp=tmp_path/'manifest.json'; mp.write_text(json.dumps(manifest),encoding='utf-8')
    return meta,split,mp,cqt_dir,rhythm_dir,ids,labels


def test_align_uses_centers_crops_common_time_and_masks_invalid():
    cqt=np.tile(np.arange(31,dtype=float),(88,1))
    starts=np.array([0.,2.,4.]); rhythm=np.tile(np.array([0.,2.,4.]),(9,1))
    rhythm[4:7,1]=0.; rhythm[0,1]=0.; rhythm[1,1]=0.; rhythm[7,1]=120.
    cfg={'cqt_fps':1.,'cqt_time_offset_seconds':0.}
    values,valid=align_arrays(cqt,rhythm,starts,4.,cfg)
    # Centers are 2,4,6: no extrapolation outside that range.
    assert values.shape==(97,5)
    np.testing.assert_array_equal(values[0],np.arange(2,7))
    np.testing.assert_allclose(values[88],np.array([0,0,0,2,4]))
    assert not valid[92:95,1:4].any()
    assert not valid[95,1:4].any()


def test_tail_window_end_times_and_short_piece_are_explicit():
    cqt=np.ones((88,20)); rhythm=np.ones((9,1)); starts=np.array([0.])
    cfg={'cqt_fps':5.,'cqt_time_offset_seconds':0.}
    with pytest.raises(ValueError,match='Single rhythm window'):
        align_arrays(cqt,rhythm,starts,4.,cfg)
    values,_=align_arrays(cqt,rhythm,starts,4.,cfg,end_times=np.array([.8]))
    assert values.shape[1]==1


def test_source_hashes_normalization_and_both_dataset_modes(tmp_path):
    meta,split,manifest,cqt_dir,rhythm_dir,ids,labels=_fixture(tmp_path)
    source=AlignedFeatureSource(ids,labels,cqt_dir,rhythm_dir,meta,split,5.,0.,manifest)
    mean,std,counts,frames=compute_aligned_statistics(source,ids.tolist())
    assert mean.shape==std.shape==counts.shape==(97,) and frames>0
    norm=tmp_path/'norm.npz'
    np.savez(norm,mean=mean,std=std,valid_counts=counts,
        preprocessing_json=np.asarray(json.dumps(source.config,sort_keys=True)),
        source_signature=np.asarray(source.source_signature),split_sha256=np.asarray(sha256(split)),
        metadata_sha256=np.asarray(sha256(meta)),train_ids_sha256=np.asarray(hashlib.sha256('piece'.encode()).hexdigest()))
    got_mean,got_std=load_aligned_normalization(norm,source,split,meta,['piece'])
    np.testing.assert_allclose(got_mean,mean); np.testing.assert_allclose(got_std,std)
    cqt_ds=AlignedSequenceDataset(ids,labels,source,mean,std,False)
    fused_ds=AlignedSequenceDataset(ids,labels,source,mean,std,True)
    assert cqt_ds[0][0].shape[0]==88 and fused_ds[0][0].shape[0]==97
    assert torch.isfinite(fused_ds[0][0]).all()
    # Any cache mutation is rejected against the manifest/source signature.
    with (cqt_dir/'piece.bin').open('ab') as f: f.write(b'x')
    with pytest.raises(ValueError,match='changed since freeze'):
        source.read('piece')


def test_crnn_accepts_88_and_97_channels_and_rejects_mixed_batch():
    for channels in (88,97):
        batch=[(torch.randn(channels,20),torch.tensor(1.),'a'),
               (torch.randn(channels,25),torch.tensor(2.),'b')]
        x,lengths,y,ids=collate_sequences(batch)
        model=CQTCRNN(conv_channels=4,hidden_size=3,dropout=0.,input_channels=channels)
        score=model(x,lengths); assert score.shape==(2,)
        ((score-y)**2).mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    with pytest.raises(ValueError,match='same channel'):
        collate_sequences([(torch.ones(88,3),torch.tensor(1.),'a'),
                           (torch.ones(97,3),torch.tensor(2.),'b')])


def test_statistics_only_reads_requested_train_ids_and_valid_values():
    class Source:
        def __init__(self): self.read_ids=[]
        def read(self,key):
            self.read_ids.append(key)
            assert key=='train', 'Validation/test must not be read for statistics'
            x=np.tile([1.,3.,1000.],(97,1)); valid=np.ones_like(x,dtype=bool)
            valid[-1,-1]=False
            valid[-2]=False
            return x,valid
    source=Source()
    mean,std,counts,frames=compute_aligned_statistics(source,['train'])
    assert source.read_ids==['train'] and frames==3
    assert mean[-1]==2 and std[-1]==1 and counts[-1]==2
    assert mean[-2]==0 and std[-2]==1 and counts[-2]==0


def test_nfc_lookup_and_invalid_alignment_config(tmp_path):
    from data.aligned_sequences import scan_files, alignment_config
    key='Café'
    decomposed=unicodedata.normalize('NFD',key)
    (tmp_path/(decomposed+'.npz')).write_bytes(b'x')
    assert key in scan_files(tmp_path,'.npz')
    for fps in (0,-1,float('nan'),float('inf')):
        with pytest.raises(ValueError): alignment_config(fps)


def test_metadata_only_accepts_verified_line_ending_changes(tmp_path):
    from data.aligned_sequences import check_metadata_hash
    p=tmp_path/'meta.json'; lf=b'{\n  "a": 1\n}\n'
    p.write_bytes(lf.replace(b'\n',b'\r\n'))
    check_metadata_hash(p,hashlib.sha256(lf).hexdigest())
    p.write_bytes(lf.replace(b'1',b'2'))
    with pytest.raises(ValueError,match='metadata mismatch'):
        check_metadata_hash(p,hashlib.sha256(lf).hexdigest())


def test_irregular_tail_is_interpolated_by_time_not_column_number():
    cqt=np.tile(np.arange(30,dtype=float),(88,1))
    starts=np.array([0.,.2,.26])
    rhythm=np.tile(np.array([1.,2.,3.]),(9,1))
    x,_=align_arrays(cqt,rhythm,starts,4.,{'cqt_fps':20.,'cqt_time_offset_seconds':2.})
    np.testing.assert_allclose(x[88],[1.,1.25,1.5,1.75,2.,2.+5./6.])
    assert x.shape[1]==6


def test_roundoff_exact_center_does_not_inherit_neighbor_invalidity():
    cqt=np.ones((88,30)); starts=np.array([0.,.2,.4])
    rhythm=np.ones((9,3));rhythm[4,2]=0
    _,valid=align_arrays(cqt,rhythm,starts,4.,
        {'cqt_fps':5.,'cqt_time_offset_seconds':1e-12})
    assert valid[92,1]  # Almost-exact 2.2 belongs to middle, not the next invalid window.


def test_real_cli_prepare_both_modes_and_checkpoint_roundtrip(tmp_path):
    meta,split,manifest,cqt_dir,rhythm_dir,ids,labels=_fixture(tmp_path,('a','b','c'))
    # Extra metadata/CQT not in frozen split must not be selected.
    md=json.loads(meta.read_text()); md['excluded']={'ps':'8','composer':'other','youtube_link':''}
    meta.write_text(json.dumps(md,indent=2)+'\n')
    sd=json.loads(split.read_text()); sd['metadata_sha256']=sha256(meta); split.write_text(json.dumps(sd))
    with np.load(rhythm_dir/'index.npz',allow_pickle=False) as index:
        idx={k:index[k].copy() for k in index.files}
    idx['metadata_sha256']=np.asarray(sha256(meta)); np.savez(rhythm_dir/'index.npz',**idx)
    mf=json.loads(manifest.read_text()); mf['split_sha256']=sha256(split)
    mf['source_sha256']['datasets/features/rhythm_windows/index.npz']=sha256(rhythm_dir/'index.npz')
    manifest.write_text(json.dumps(mf))
    args=['--metadata',str(meta),'--split',str(split),'--cqt-dir',str(cqt_dir)]
    aligned_args=args+['--rhythm-dir',str(rhythm_dir),'--manifest',str(manifest)]
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    def run(script,arguments,ok=True):
        result=subprocess.run([sys.executable,'-B','-W','error',str(ROOT/'src'/script),*arguments],
            env=env,capture_output=True,text=True,timeout=120)
        assert (result.returncode==0)==ok, result.stdout+'\n'+result.stderr
        return result
    norm=tmp_path/'aligned_norm.npz'
    run('prepare_crnn.py',aligned_args+['--output',str(norm)])
    with np.load(norm,allow_pickle=False) as d:
        mean,std=d['mean'].copy(),d['std'].copy()
        assert mean.shape==(97,) and d['n_frames']==30
    source=AlignedFeatureSource(ids,labels,cqt_dir,rhythm_dir,meta,split,manifest_path=manifest)
    ds0=AlignedSequenceDataset(['b'],[3],source,mean,std,False)
    ds1=AlignedSequenceDataset(['b'],[3],source,mean,std,True)
    torch.testing.assert_close(ds0[0][0],ds1[0][0][:88],rtol=0,atol=0)
    for mode,ds,channels in [('cqt',ds0,88),('cqt-rhythm',ds1,97)]:
        output=tmp_path/('run_'+mode)
        command=aligned_args+['--normalization',str(norm),'--feature-mode',mode,
            '--device','cpu','--num-workers','0','--batch-size','2','--max-epochs','1',
            '--patience','1','--conv-channels','4','--hidden-size','3','--dropout','0',
            '--no-amp','--output',str(output)]
        run('train_crnn.py',command)
        result=json.loads((output/'metrics.json').read_text())
        saved_config=json.loads((output/'config.json').read_text())
        assert saved_config['training']['epochs_run']==1
        assert saved_config['training']['min_delta']==0.001
        # Constant one-piece validation has undefined Tau-c and consumes patience.
        assert saved_config['training']['stop_reason']=='early_stopping'
        stream=[json.loads(line) for line in (output/'history.jsonl').read_text().splitlines()]
        assert stream==json.loads((output/'history.json').read_text())
        assert json.loads((output/'run_parameters.json').read_text())['feature_mode']==mode
        assert result['test_evaluated'] is False and 'test' not in result
        assert not (output/'predictions_test.json').exists()
        artifact=torch.load(output/'model.pt',map_location='cpu',weights_only=True)
        assert artifact['config']['sizes']=={'train':1,'val':1,'test':1}
        assert artifact['config']['architecture']['input_bins']==channels
        model=CQTCRNN(4,3,0,input_channels=channels); model.load_state_dict(artifact['state_dict']);model.eval()
        x,lengths,_,_=collate_sequences([ds[0]])
        with torch.no_grad(): pred=model(x,lengths).item()
        rows=json.loads((output/'predictions_val.json').read_text())
        assert rows[0]['id']=='b' and rows[0]['score']==pytest.approx(pred,abs=1e-5)
        run('train_crnn.py',command,ok=False)  # No overwrite.
    # Preparation must not change frozen files/data or use validation/test values.
    assert sha256(split)==mf['split_sha256']
    run('prepare_crnn.py',aligned_args+['--output',str(norm)],ok=False)
    bad_norm=tmp_path/'wrong_offset.npz'
    run('prepare_crnn.py',aligned_args+['--cqt-time-offset','.1','--output',str(bad_norm)])
    with pytest.raises(ValueError,match='configuration mismatch'):
        load_aligned_normalization(bad_norm,source,split,meta,['a'])
    # Legacy 88-bin mode still works on a subset without rhythm input.
    legacy=tmp_path/'legacy.npz'
    run('prepare_crnn.py',args+['--output',str(legacy)])
    run('train_crnn.py',args+['--normalization',str(legacy),'--device','cpu',
        '--num-workers','0','--batch-size','2','--max-epochs','1','--patience','1',
        '--conv-channels','4','--hidden-size','3','--no-amp','--output',str(tmp_path/'legacy_run')])

