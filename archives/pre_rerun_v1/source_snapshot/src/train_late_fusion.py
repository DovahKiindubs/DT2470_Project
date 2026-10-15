"""Train native-resolution CQT/rhythm two-branch CRNN ablations and late fusion."""
import argparse
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from data.aligned_sequences import load_split
from data.native_sequences import (NativePairSource,NativeTwoBranchDataset,
                                   collate_native_pairs,load_native_normalization)
from evaluate import evaluate
from features_cqt import ROOT,sha256
from models.model_crnn import LengthBucketBatchSampler
from models.model_late_fusion import LateFusionCRNN
from train_crnn import EarlyStopping,choose_device,write_json


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True


def move_batch(batch,device):
    c,cl,r,rl,y,ids=batch
    nonblocking=device.type=='cuda'
    return c.to(device,non_blocking=nonblocking),cl,r.to(device,non_blocking=nonblocking),rl,y.to(device,non_blocking=nonblocking),ids


def predict(model,loader,device,amp_enabled=False):
    model.eval();ids=[];truth=[];scores=[]
    with torch.inference_mode():
        for batch in loader:
            c,cl,r,rl,y,batch_ids=move_batch(batch,device)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp_enabled):
                pred=model(c,cl,r,rl)
            ids.extend(batch_ids);truth.extend(y.cpu().numpy());scores.extend(pred.float().cpu().numpy())
    return ids,np.asarray(truth),np.asarray(scores)


def main(args):
    if args.output.exists():raise FileExistsError(args.output)
    positive=(args.batch_size,args.max_epochs,args.patience,args.cqt_conv_channels,
              args.cqt_hidden_size,args.rhythm_conv_channels,args.rhythm_hidden_size,args.embedding_size)
    if min(positive)<=0:raise ValueError('Positive hyperparameters required')
    if args.num_workers<0 or args.prefetch_factor<=0:raise ValueError('Invalid worker settings')
    if (not np.isfinite([args.learning_rate,args.weight_decay,args.grad_clip,args.dropout,args.min_delta]).all()
            or args.learning_rate<=0 or args.weight_decay<0 or args.grad_clip<=0
            or not 0<=args.dropout<1 or args.min_delta<0):
        raise ValueError('Invalid training hyperparameters')
    stopping=EarlyStopping(args.patience,args.min_delta);seed_all(args.seed)
    split_doc,metadata,ids,labels,indices=load_split(args.metadata,args.split)
    source=NativePairSource(ids,labels,args.cqt_dir,args.rhythm_dir,args.metadata,args.split,args.manifest)
    cm,cs,rm,rs=load_native_normalization(args.normalization,source,args.split,args.metadata,
                                           split_doc['parts']['train'])
    names=('train','val','test') if args.evaluate_test else ('train','val')
    datasets={name:NativeTwoBranchDataset(ids[indices[name]],labels[indices[name]],source,cm,cs,rm,rs)
              for name in names}
    sampler=LengthBucketBatchSampler(datasets['train'].lengths,args.batch_size,args.seed,True)
    device=choose_device(args.device)
    loader_kwargs={'collate_fn':collate_native_pairs,'num_workers':args.num_workers,
                   'pin_memory':args.pin_memory and device.type=='cuda'}
    if args.num_workers>0:
        loader_kwargs['persistent_workers']=True;loader_kwargs['prefetch_factor']=args.prefetch_factor
    loaders={'train':DataLoader(datasets['train'],batch_sampler=sampler,**loader_kwargs),
             'val':DataLoader(datasets['val'],batch_sampler=LengthBucketBatchSampler(
                 datasets['val'].lengths,args.batch_size,args.seed,False),**loader_kwargs)}
    if args.evaluate_test:
        loaders['test']=DataLoader(datasets['test'],batch_sampler=LengthBucketBatchSampler(
            datasets['test'].lengths,args.batch_size,args.seed,False),**loader_kwargs)
    model=LateFusionCRNN(args.mode,args.cqt_conv_channels,args.cqt_hidden_size,
                         args.rhythm_conv_channels,args.rhythm_hidden_size,
                         args.embedding_size,args.dropout).to(device)
    amp_enabled=args.amp and device.type=='cuda';scaler=torch.amp.GradScaler('cuda',enabled=amp_enabled)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)
    loss_fn=nn.MSELoss();args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'run_parameters.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    best=None;history=[];start=time.time();stop_reason='max_epochs';best_path=args.output/'best_checkpoint.pt'
    for epoch in range(1,args.max_epochs+1):
        sampler.set_epoch(epoch);model.train();total=0.;count=0
        for batch in loaders['train']:
            c,cl,r,rl,y,_=move_batch(batch,device);optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp_enabled):
                pred=model(c,cl,r,rl);loss=loss_fn(pred,y)
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(),args.grad_clip)
            scaler.step(optimizer);scaler.update();total+=float(loss.detach().float().cpu())*len(y);count+=len(y)
        _,vy,vp=predict(model,loaders['val'],device,amp_enabled);metrics=evaluate(vy,vp);tau=metrics['kendall_tau_c']
        key=(-np.inf if tau is None else tau,-metrics['mse'])
        if best is None or key>best['key']:
            best={'key':key,'epoch':epoch,'state':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'metrics':metrics}
            torch.save({'state_dict':best['state'],'epoch':epoch,'metrics':metrics,
                        'normalization_sha256':sha256(args.normalization)},best_path)
        should_stop=stopping.step(tau)
        row={'epoch':epoch,'train_mse':total/count,'validation':metrics,
             'early_stopping':{'stale_epochs':stopping.stale_epochs,'patience':args.patience,
             'min_delta':args.min_delta,'reference_tau':stopping.reference_tau},'best_epoch':best['epoch']}
        history.append(row)
        with (args.output/'history.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        print(json.dumps(row,ensure_ascii=False,allow_nan=False),flush=True)
        if should_stop:stop_reason='early_stopping';break
    model.load_state_dict(best['state']);model.to(device)
    predictions={};results={'validation':best['metrics'],'test_evaluated':args.evaluate_test}
    for name in (('val','test') if args.evaluate_test else ('val',)):
        keys,y,pred=predict(model,loaders[name],device,amp_enabled)
        predictions[name]=[{'id':k,'true_grade':int(t),'score':float(v)} for k,t,v in zip(keys,y,pred)]
        if name=='test':results['test']=evaluate(y,pred)
    config={'feature_group':'native_two_branch_'+args.mode.replace('-','_'),
      'model':'Independent CQT/rhythm Conv1d-BiGRU-attention branches; equal-size LayerNorm projections; concatenation + learned MLP',
      'architecture':{'fusion':'representation concatenation','fixed_feature_weights':False,
        'cqt':{'input_channels':88,'conv_channels':args.cqt_conv_channels,'hidden_per_direction':args.cqt_hidden_size},
        'rhythm':{'input_channels':9,'conv_channels':args.rhythm_conv_channels,'hidden_per_direction':args.rhythm_hidden_size},
        'embedding_size_per_branch':args.embedding_size,'dropout':args.dropout},
      'training':{'loss':'MSE','optimizer':'AdamW','learning_rate':args.learning_rate,
        'weight_decay':args.weight_decay,'batch_size':args.batch_size,'max_epochs':args.max_epochs,
        'patience':args.patience,'min_delta':args.min_delta,'epochs_run':len(history),
        'stop_reason':stop_reason,'selection':'validation Tau-c, then lower validation MSE',
        'best_epoch':best['epoch'],'seed':args.seed,'amp':amp_enabled,'num_workers':args.num_workers},
      'native_time_axes':True,'rhythm_interpolated_to_cqt':False,
      'normalization':'Separate train-only channel mean/std; invalid rhythm observations -> standardized 0',
      'source_signature':source.source_signature,'manifest_sha256':source.manifest_sha256,
      'split_file':str(args.split.resolve()),'split_sha256':sha256(args.split),
      'normalization_sha256':sha256(args.normalization),'sizes':{k:len(v) for k,v in indices.items()},
      'device':str(device),'runtime_seconds':time.time()-start,'python':platform.python_version(),
      'torch':str(torch.__version__),'test_evaluated':args.evaluate_test,
      'limitations':['No fixed feature weights; learned fusion head may still prefer one branch.',
                     'Single frozen split unless repeated with additional seeds.',
                     'Rhythm validity uses heuristic masks; tempo_confidence is not calibrated.',
                     'Original convolution padding/BatchNorm behavior remains unchanged.',
                     'No test evaluation unless explicitly requested.']}
    write_json(args.output/'config.json',config);write_json(args.output/'metrics.json',results)
    write_json(args.output/'history.json',history)
    for name,rows in predictions.items():write_json(args.output/f'predictions_{name}.json',rows)
    torch.save({'state_dict':best['state'],'config':config,
                'normalization_sha256':sha256(args.normalization)},args.output/'model.pt')
    print(json.dumps({'output':str(args.output),'best_epoch':best['epoch'],'results':results},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['cqt-only','rhythm-only','late-fusion'],default='late-fusion')
    p.add_argument('--cqt-dir',type=Path,default=ROOT/'datasets/features/cqt')
    p.add_argument('--rhythm-dir',type=Path,default=ROOT/'datasets/features/rhythm_windows')
    p.add_argument('--metadata',type=Path,default=ROOT/'datasets/metadata/new_clean_data.json')
    p.add_argument('--split',type=Path,default=ROOT/'datasets/splits/cqt_rhythm_common_v1.json')
    p.add_argument('--manifest',type=Path,default=None)
    p.add_argument('--normalization',type=Path,default=ROOT/'datasets/features/cqt_rhythm_native_normalization.npz')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/cqt_rhythm_common_v1/late_fusion/run_01')
    p.add_argument('--device',choices=['auto','cpu','mps','cuda'],default='auto')
    p.add_argument('--seed',type=int,default=42);p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--num-workers',type=int,default=4);p.add_argument('--prefetch-factor',type=int,default=2)
    p.add_argument('--pin-memory',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--amp',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--cqt-conv-channels',type=int,default=64);p.add_argument('--cqt-hidden-size',type=int,default=64)
    p.add_argument('--rhythm-conv-channels',type=int,default=32);p.add_argument('--rhythm-hidden-size',type=int,default=32)
    p.add_argument('--embedding-size',type=int,default=64);p.add_argument('--dropout',type=float,default=.2)
    p.add_argument('--learning-rate',type=float,default=1e-3);p.add_argument('--weight-decay',type=float,default=1e-4)
    p.add_argument('--grad-clip',type=float,default=1.);p.add_argument('--max-epochs',type=int,default=100)
    p.add_argument('--patience',type=int,default=15);p.add_argument('--min-delta',type=float,default=.001)
    p.add_argument('--evaluate-test',action='store_true')
    main(p.parse_args())
