"""One-shot test-only evaluation for the nine frozen rerun_v1 models.

This module never fits, tunes, selects, or rewrites a model. It loads the exact
saved artifacts, scores the frozen test partition, and writes a separate release.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader

from data.aligned_sequences import (AlignedFeatureSource, AlignedSequenceDataset,
                                    load_aligned_normalization, load_split)
from data.feature_matrix import prepare_features
from data.native_sequences import (NativePairSource, NativeTwoBranchDataset,
                                    collate_native_pairs, load_native_normalization)
from experiments.suite import (CATALOGUE, ROOT, now, plan, preflight, read_json,
                               select, sha, verify_outputs, verify_result, write_json)
from models.model_crnn import CQTCRNN, LengthBucketBatchSampler, collate_sequences
from models.model_late_fusion import LateFusionCRNN
from reporting.metrics import evaluate
from training.crnn import choose_device, predict as predict_early
from training.late_fusion import predict as predict_late

ACKNOWLEDGEMENT = 'FROZEN_MODELS_FINAL_TEST'


def _git_revision(code_root):
    result = subprocess.run(['git','-C',str(code_root),'rev-parse','HEAD'],
                            capture_output=True,text=True,check=True)
    return result.stdout.strip()


def _source_hashes(code_root):
    return {str(p.relative_to(code_root)):sha(p) for p in sorted((Path(code_root)/'src').rglob('*.py'))}


def _recorded_sources_match(state, code_root):
    for name, expected in state.get('source_sha256',{}).items():
        path=Path(code_root)/name
        if not path.is_file() or sha(path)!=expected:
            raise ValueError(f'Training source changed or is missing: {name}')


def _frozen_runs(catalogue, data_root, code_root, seed=42, run_name='run_01'):
    rows=select(catalogue)
    verification_catalogue=dict(catalogue,
        metadata=str(Path(data_root)/catalogue['metadata']),
        split=str(Path(data_root)/catalogue['split']))
    verification_catalogue['inputs']=dict(catalogue['inputs'])
    verification_catalogue['inputs'][verification_catalogue['split']]=catalogue['inputs'][catalogue['split']]
    items=[]
    for row in rows:
        item=plan(catalogue,row,seed,run_name,root=data_root)
        run=item['run'];state_path=run/'run.json'
        if not state_path.is_file():raise FileNotFoundError(f'Missing run state: {state_path}')
        state=read_json(state_path)
        if state.get('status')!='completed':raise ValueError(f'Run is not completed: {run}')
        verify_outputs(item);verify_result(run,verification_catalogue,seed);_recorded_sources_match(state,code_root)
        items.append((row,item,state))
    return items


def _rows(ids, y, scores):
    return [{'id':str(k),'true_grade':int(t),'score':float(v)} for k,t,v in zip(ids,y,scores)]


def _verify_test_rows(rows, split, metadata):
    keys=[r['id'] for r in rows]
    if len(keys)!=len(set(keys)) or set(keys)!=set(split['parts']['test']):
        raise ValueError('Predictions do not match the frozen test partition')
    if any(int(metadata[r['id']]['ps'])!=r['true_grade'] for r in rows):
        raise ValueError('Test prediction labels disagree with metadata')
    return evaluate([r['true_grade'] for r in rows],[r['score'] for r in rows])


def _test_hgb(row, item, context):
    catalogue,root,metadata,ids,labels,indices=context
    args=catalogue['groups']['hgb']['args']+row['args']
    main=root/args[args.index('--features')+1]
    extras=[root/args[args.index(flag)+1] for flag in ('--rhythm-features','--structure-features') if flag in args]
    x,y,feature_ids,names,_=prepare_features(main,metadata,list(ids),extras)
    if not np.array_equal(y,labels):raise ValueError('HGB labels changed')
    artifact=joblib.load(item['run']/'artifacts/model.joblib')
    if artifact.get('feature_names')!=names:raise ValueError('HGB feature names changed')
    ix=indices['test'];scores=artifact['model'].predict(x[ix])
    return _rows(feature_ids[ix],y[ix],scores)


def _early_context(catalogue, root, split, metadata, ids, labels, indices,
                   batch_size, workers, device):
    source=AlignedFeatureSource(ids,labels,root/'datasets/features/cqt',root/'datasets/features/rhythm_windows',
        root/catalogue['metadata'],root/catalogue['split'],5.,0.,root/catalogue['manifest'])
    norm=root/'datasets/features/cqt_rhythm_common_v1_normalization.npz'
    mean,std=load_aligned_normalization(norm,source,root/catalogue['split'],root/catalogue['metadata'],split['parts']['train'])
    kwargs={'collate_fn':collate_sequences,'num_workers':workers,'pin_memory':device.type=='cuda'}
    if workers>0:kwargs.update(persistent_workers=True,prefetch_factor=2)
    return source,mean,std,kwargs,norm


def _test_early(row,item,context):
    source,mean,std,kwargs,norm,ids,labels,indices,batch_size,device=context
    include=row['id']=='early_cqt_rhythm'
    dataset=AlignedSequenceDataset(ids[indices['test']],labels[indices['test']],source,mean,std,include)
    loader=DataLoader(dataset,batch_sampler=LengthBucketBatchSampler(dataset.lengths,batch_size,42,False),**kwargs)
    config=read_json(item['run']/'artifacts/config.json');arch=config['architecture']
    artifact=torch.load(item['run']/'artifacts/model.pt',map_location='cpu',weights_only=True)
    if artifact.get('config')!=config or artifact.get('normalization_sha256')!=sha(norm):
        raise ValueError('Early model/config/normalization mismatch')
    model=CQTCRNN(arch['conv_channels'],arch['gru_hidden_size_per_direction'],arch['dropout'],
                  input_channels=arch['input_bins']).to(device)
    model.load_state_dict(artifact['state_dict'])
    batch_ids,y,scores=predict_early(model,loader,device,device.type=='cuda' and config['training']['amp'])
    del model,loader,dataset
    if device.type=='cuda':torch.cuda.empty_cache()
    return _rows(batch_ids,y,scores)


def _late_context(catalogue,root,split,metadata,ids,labels,indices,batch_size,workers,device):
    source=NativePairSource(ids,labels,root/'datasets/features/cqt',root/'datasets/features/rhythm_windows',
        root/catalogue['metadata'],root/catalogue['split'],root/catalogue['manifest'])
    norm=root/'datasets/features/cqt_rhythm_native_normalization.npz'
    cm,cs,rm,rs=load_native_normalization(norm,source,root/catalogue['split'],root/catalogue['metadata'],split['parts']['train'])
    dataset=NativeTwoBranchDataset(ids[indices['test']],labels[indices['test']],source,cm,cs,rm,rs)
    kwargs={'collate_fn':collate_native_pairs,'num_workers':workers,'pin_memory':device.type=='cuda'}
    if workers>0:kwargs.update(persistent_workers=True,prefetch_factor=2)
    loader=DataLoader(dataset,batch_sampler=LengthBucketBatchSampler(dataset.lengths,batch_size,42,False),**kwargs)
    return loader,norm


def _test_late(row,item,context):
    loader,norm,device=context
    config=read_json(item['run']/'artifacts/config.json');arch=config['architecture']
    artifact=torch.load(item['run']/'artifacts/model.pt',map_location='cpu',weights_only=True)
    if artifact.get('config')!=config or artifact.get('normalization_sha256')!=sha(norm):
        raise ValueError('Late model/config/normalization mismatch')
    mode=row['args'][row['args'].index('--mode')+1]
    model=LateFusionCRNN(mode,arch['cqt']['conv_channels'],arch['cqt']['hidden_per_direction'],
        arch['rhythm']['conv_channels'],arch['rhythm']['hidden_per_direction'],
        arch['embedding_size_per_branch'],arch['dropout']).to(device)
    model.load_state_dict(artifact['state_dict'])
    batch_ids,y,scores=predict_late(model,loader,device,device.type=='cuda' and config['training']['amp'])
    del model
    if device.type=='cuda':torch.cuda.empty_cache()
    return _rows(batch_ids,y,scores)


def _summary(output,catalogue):
    state=read_json(output/'release.json');rows=[]
    for row in catalogue['experiments']:
        path=output/row['group']/row['folder']/'metrics.json'
        metrics=read_json(path)['test'] if path.is_file() else None
        rows.append({'experiment_id':row['id'],'group':row['group'],'status':'completed' if metrics else 'pending','test':metrics})
    write_json(output/'results.json',{'release':state,'runs':rows})
    lines=['# Final frozen test evaluation','',
           'These metrics come from the nine pre-frozen saved models. No fitting or model selection was performed.','']
    for group in catalogue['groups']:
        lines.extend(['## '+group,''])
        for result in (x for x in rows if x['group']==group):
            text=f'- **{result["experiment_id"]}** — {result["status"]}'
            if result['test']:
                text+=' · '+', '.join(f'{k}={result["test"][k]:.4f}' for k in ('kendall_tau_c','mae','mse','acc_pm1'))
            lines.append(text)
        lines.append('')
    (output/'RESULTS.md').write_text('\n'.join(lines).rstrip()+'\n',encoding='utf-8')


def evaluate_release(catalogue, data_root=ROOT, code_root=ROOT, output=None, seed=42,
                     run_name='run_01', device_name='cuda', batch_size=32, workers=4,
                     resume=False):
    data_root,code_root=Path(data_root),Path(code_root)
    output=Path(output or data_root/'outputs/rerun_v1/test_evaluation/final_01')
    lock_path=output.parent/(output.name+'.lock');lock_path.parent.mkdir(parents=True,exist_ok=True)
    with lock_path.open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:raise RuntimeError('Another final-test process is running') from exc
        if output.exists():
            state=read_json(output/'release.json') if (output/'release.json').is_file() else {}
            if state.get('status')=='completed':raise FileExistsError('Final test already completed; refusing a second look')
            if not resume:raise FileExistsError('Partial final test exists; inspect it and pass --resume')
        else:
            output.mkdir(parents=True)
            state={'status':'running','acknowledged_final_test':True,'started_at_utc':now(),
                   'seed':seed,'run_name':run_name,'catalogue_sha256':sha(CATALOGUE),
                   'git_revision':_git_revision(code_root),'python':platform.python_version(),
                   'torch':str(torch.__version__),'device_requested':device_name,
                   'training_performed':False,'model_selection_performed':False}
            write_json(output/'release.json',state,exclusive=True)
        items=_frozen_runs(catalogue,data_root,code_root,seed,run_name)
        provenance=preflight(catalogue,[r for r,_,_ in items],root=data_root)
        signature={'catalogue_sha256':sha(CATALOGUE),'git_revision':_git_revision(code_root),
                   'source_sha256':_source_hashes(code_root),
                   'models':{r['id']:sha(i['run']/'artifacts/model.joblib' if r['group']=='hgb' else i['run']/'artifacts/model.pt') for r,i,_ in items},
                   'inputs':provenance['input_sha256']}
        if state.get('signature') and state['signature']!=signature:raise ValueError('Data/model/code changed since partial test')
        state['signature']=signature;write_json(output/'release.json',state)
        split,metadata,ids,labels,indices=load_split(data_root/catalogue['metadata'],data_root/catalogue['split'])
        device=choose_device(device_name)
        early=None;late=None
        started=time.time()
        try:
            for row,item,_ in items:
                target=output/row['group']/row['folder'];metrics_path=target/'metrics.json'
                if metrics_path.exists():
                    saved=read_json(target/'predictions_test.json')
                    computed=_verify_test_rows(saved,split,metadata)
                    if computed!=read_json(metrics_path)['test']:raise ValueError(f'Existing test result corrupted: {row["id"]}')
                    continue
                target.mkdir(parents=True,exist_ok=True)
                if any(target.iterdir()):
                    raise ValueError(f'Partial test directory is not empty but has no complete result: {target}')
                if row['group']=='hgb':
                    predictions=_test_hgb(row,item,(catalogue,data_root,metadata,ids,labels,indices))
                elif row['group']=='early':
                    if early is None:
                        early0=_early_context(catalogue,data_root,split,metadata,ids,labels,indices,batch_size,workers,device)
                        early=(*early0,ids,labels,indices,batch_size,device)
                    predictions=_test_early(row,item,early)
                else:
                    if late is None:
                        late0=_late_context(catalogue,data_root,split,metadata,ids,labels,indices,batch_size,workers,device)
                        late=(*late0,device)
                    predictions=_test_late(row,item,late)
                metrics=_verify_test_rows(predictions,split,metadata)
                write_json(target/'predictions_test.json',predictions,exclusive=True)
                write_json(target/'metrics.json',{'test':metrics,'test_evaluated':True,
                    'experiment_id':row['id'],'source_run':str(item['run'].relative_to(data_root)),
                    'model_sha256':signature['models'][row['id']]},exclusive=True)
                _summary(output,catalogue)
            state.update(status='completed',finished_at_utc=now(),runtime_seconds=time.time()-started,
                         device=str(device),training_performed=False,model_selection_performed=False)
            write_json(output/'release.json',state);_summary(output,catalogue)
        except BaseException as exc:
            state.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',
                         finished_at_utc=now(),error=f'{type(exc).__name__}: {exc}',
                         training_performed=False,model_selection_performed=False)
            write_json(output/'release.json',state);_summary(output,catalogue);raise
    return read_json(output/'results.json')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--acknowledge-final-test',default='')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--device',choices=['cuda','cpu','mps','auto'],default='cuda')
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--num-workers',type=int,default=4)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--run-name',default='run_01')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/rerun_v1/test_evaluation/final_01')
    args=parser.parse_args();catalogue=read_json(CATALOGUE)
    print('This command loads all nine frozen models and scores the 1,557-piece test partition.')
    print('It does not train, tune, select, or modify any model.')
    print('Output:',args.output)
    if not args.execute:return
    if args.acknowledge_final_test!=ACKNOWLEDGEMENT:
        parser.error(f'--acknowledge-final-test must equal {ACKNOWLEDGEMENT}')
    result=evaluate_release(catalogue,output=args.output,seed=args.seed,run_name=args.run_name,
        device_name=args.device,batch_size=args.batch_size,workers=args.num_workers,resume=args.resume)
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__':main()
