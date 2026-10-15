"""训练轻量 CQT CRNN；验证集早停，默认不评估测试集。"""
import argparse
import json
import os
import platform
import random
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from reporting.metrics import evaluate
from features.cqt import ROOT, sha256
from models.model_crnn import CQTCRNN, CQTSequenceDataset, LengthBucketBatchSampler, collate_sequences
from data.split_data import make_groups, validate_split
from data.aligned_sequences import (AlignedFeatureSource, AlignedSequenceDataset,
                                    check_metadata_hash, load_aligned_normalization, load_split)


def write_json(path,data):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False); f.write('\n')


class EarlyStopping:
    """Patience tracks meaningful Tau-c gains, independently of best-checkpoint selection."""
    def __init__(self, patience=15, min_delta=0.001):
        if patience <= 0 or not np.isfinite(min_delta) or min_delta < 0:
            raise ValueError('Early stopping requires patience > 0 and finite min_delta >= 0')
        self.patience = patience
        self.min_delta = min_delta
        self.reference_tau = None
        self.stale_epochs = 0

    def step(self, tau):
        if tau is not None and not np.isfinite(tau):
            raise ValueError('Non-finite validation Tau-c')
        improved = tau is not None and (self.reference_tau is None or
                    (tau > self.reference_tau and tau - self.reference_tau >= self.min_delta))
        if improved:
            self.reference_tau = float(tau)
            self.stale_epochs = 0
        else:
            self.stale_epochs += 1
        return self.stale_epochs >= self.patience


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def choose_device(requested):
    if requested != 'auto':
        device = torch.device(requested)
        if device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
        if device.type == 'mps' and not torch.backends.mps.is_available():
            raise RuntimeError('MPS requested but unavailable')
        return device
    if torch.cuda.is_available(): return torch.device('cuda')
    if torch.backends.mps.is_available(): return torch.device('mps')
    return torch.device('cpu')


def predict(model, loader, device, amp_enabled=False):
    model.eval(); ids=[]; truth=[]; scores=[]
    with torch.inference_mode():
        for x,lengths,y,batch_ids in loader:
            x=x.to(device,non_blocking=device.type=='cuda')
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp_enabled):
                pred=model(x,lengths)
            pred=pred.float().cpu().numpy()
            ids.extend(batch_ids); truth.extend(y.numpy()); scores.extend(pred)
    return ids,np.asarray(truth),np.asarray(scores)


def main(args):
    if args.output.exists(): raise FileExistsError(args.output)
    if min(args.batch_size,args.max_epochs,args.patience,args.hidden_size,args.conv_channels) <= 0:
        raise ValueError('Invalid positive hyperparameter')
    if args.num_workers < 0 or args.prefetch_factor <= 0:
        raise ValueError('num-workers must be nonnegative and prefetch-factor positive')
    if not np.isfinite([args.learning_rate,args.weight_decay,args.grad_clip,args.dropout]).all():
        raise ValueError('Training hyperparameters must be finite')
    if args.learning_rate <= 0 or args.weight_decay < 0 or args.grad_clip <= 0 or not 0 <= args.dropout < 1:
        raise ValueError('Invalid learning rate, weight decay, gradient clip or dropout')
    stopping=EarlyStopping(args.patience,args.min_delta)
    seed_all(args.seed)
    split_doc,metadata,ids,labels,indices=load_split(args.metadata,args.split)
    aligned = args.rhythm_dir is not None
    source = None
    if args.feature_mode == 'cqt-rhythm' and not aligned:
        raise ValueError('--feature-mode cqt-rhythm requires --rhythm-dir')
    if not aligned and (args.manifest is not None or args.cqt_fps != 5.0 or args.cqt_time_offset != 0.0):
        raise ValueError('Alignment options require --rhythm-dir')
    names = ('train','val','test') if args.evaluate_test else ('train','val')
    if aligned:
        source=AlignedFeatureSource(ids,labels,args.cqt_dir,args.rhythm_dir,
            args.metadata,args.split,args.cqt_fps,args.cqt_time_offset,args.manifest)
        mean,std=load_aligned_normalization(args.normalization,source,args.split,args.metadata,
                                            split_doc['parts']['train'])
        datasets={name:AlignedSequenceDataset(ids[indices[name]],labels[indices[name]],source,
                    mean,std,args.feature_mode=='cqt-rhythm') for name in names}
    else:
        with np.load(args.normalization,allow_pickle=False) as norm:
            mean,std=norm['mean'],norm['std']
            if 'preprocessing_json' in norm.files:
                raise ValueError('Aligned normalization requires --rhythm-dir, also for the CQT control')
            if str(norm['split_sha256'].item()) != sha256(args.split):
                raise ValueError('Normalization was not computed from this split')
            check_metadata_hash(args.metadata,str(norm['metadata_sha256'].item()))
            if (mean.shape != (88,) or std.shape != (88,) or not np.isfinite(mean).all()
                    or not np.isfinite(std).all() or np.any(std <= 0)):
                raise ValueError('Invalid CQT normalization')
        datasets={name:CQTSequenceDataset(ids[indices[name]],labels[indices[name]],args.cqt_dir,mean,std)
                  for name in names}
    input_channels=97 if args.feature_mode=='cqt-rhythm' else 88
    train_sampler=LengthBucketBatchSampler(datasets['train'].lengths,args.batch_size,args.seed,True)
    device=choose_device(args.device)
    loader_kwargs={'collate_fn':collate_sequences,'num_workers':args.num_workers,
                   'pin_memory':args.pin_memory and device.type=='cuda'}
    if args.num_workers>0:
        loader_kwargs['persistent_workers']=True
        loader_kwargs['prefetch_factor']=args.prefetch_factor
    loaders={
        'train':DataLoader(datasets['train'],batch_sampler=train_sampler,**loader_kwargs),
        'val':DataLoader(datasets['val'],batch_sampler=LengthBucketBatchSampler(
            datasets['val'].lengths,args.batch_size,args.seed,False),**loader_kwargs),
    }
    if args.evaluate_test:
        loaders['test']=DataLoader(datasets['test'],batch_sampler=LengthBucketBatchSampler(
            datasets['test'].lengths,args.batch_size,args.seed,False),**loader_kwargs)
    model=CQTCRNN(args.conv_channels,args.hidden_size,args.dropout,input_channels=input_channels).to(device)
    amp_enabled=args.amp and device.type=='cuda'
    scaler=torch.amp.GradScaler('cuda',enabled=amp_enabled)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)
    loss_fn=nn.MSELoss()
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'run_parameters.json',
               {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    best_path=args.output/'best_checkpoint.pt'
    best=None; history=[]; start=time.time(); stop_reason='max_epochs'
    for epoch in range(1,args.max_epochs+1):
        train_sampler.set_epoch(epoch); model.train(); total=0.; count=0
        for x,lengths,y,_ in loaders['train']:
            x=x.to(device,non_blocking=device.type=='cuda')
            y=y.to(device,non_blocking=device.type=='cuda')
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp_enabled):
                pred=model(x,lengths); loss=loss_fn(pred,y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(),args.grad_clip)
            scaler.step(optimizer); scaler.update()
            total += float(loss.detach().float().cpu())*len(y); count += len(y)
        _,val_y,val_pred=predict(model,loaders['val'],device,amp_enabled)
        metrics=evaluate(val_y,val_pred); tau=metrics['kendall_tau_c']
        key=(-np.inf if tau is None else tau,-metrics['mse'])
        if best is None or key>best['key']:
            best={'key':key,'epoch':epoch,'state':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'metrics':metrics}
            torch.save({'state_dict':best['state'],'epoch':epoch,'metrics':metrics,
                        'normalization_sha256':sha256(args.normalization)},best_path)
        should_stop=stopping.step(tau)
        row={'epoch':epoch,'train_mse':total/count,'validation':metrics,
             'early_stopping':{'stale_epochs':stopping.stale_epochs,'patience':args.patience,
                               'min_delta':args.min_delta,'reference_tau':stopping.reference_tau},
             'best_epoch':best['epoch']}
        history.append(row)
        with (args.output/'history.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        print(json.dumps(row,ensure_ascii=False,allow_nan=False),flush=True)
        if should_stop:
            stop_reason='early_stopping'
            break
    print(json.dumps({'stop_reason':stop_reason,'epochs_run':len(history),
                      'best_epoch':best['epoch'],'best_tau_c':best['metrics']['kendall_tau_c']}),flush=True)
    model.load_state_dict(best['state']); model.to(device)
    predictions={}; results={'validation':best['metrics'],'test_evaluated':args.evaluate_test}
    for name in (('val','test') if args.evaluate_test else ('val',)):
        batch_ids,y,pred=predict(model,loaders[name],device,amp_enabled)
        predictions[name]=[{'id':key,'true_grade':int(t),'score':float(p)} for key,t,p in zip(batch_ids,y,pred)]
        if name=='test': results['test']=evaluate(y,pred)
    config={'feature_group':('cqt_rhythm_sequence' if input_channels==97 else
                             'cqt_sequence_aligned' if aligned else 'cqt_sequence'),
        'model':'Conv1d x2 + BiGRU + masked attention + linear head',
        'architecture':{'input_bins':input_channels,'conv_channels':args.conv_channels,'conv_kernel':5,'conv_time_stride':4,
          'gru_hidden_size_per_direction':args.hidden_size,'bidirectional':True,'dropout':args.dropout},
        'training':{'loss':'MSE','optimizer':'AdamW','learning_rate':args.learning_rate,
          'weight_decay':args.weight_decay,'batch_size':args.batch_size,'max_epochs':args.max_epochs,
          'patience':args.patience,'min_delta':args.min_delta,
          'early_stopping_monitor':'validation Tau-c; cumulative gain from last reset >= min_delta',
          'epochs_run':len(history),'stop_reason':stop_reason,
          'gradient_clip':args.grad_clip,'seed':args.seed,
          'num_workers':args.num_workers,'pin_memory':args.pin_memory,'amp':amp_enabled,
          'selection':'validation Tau-c, then lower validation MSE','best_epoch':best['epoch']},
        'normalization':('Per-channel valid aligned train-frame mean/std; invalid standardized values -> 0'
                         if aligned else 'Per-CQT-bin mean/std computed across all train frames only'),
        'feature_mode':args.feature_mode,
        'preprocessing':source.config if aligned else None,
        'source_signature':source.source_signature if aligned else None,
        'manifest_sha256':source.manifest_sha256 if aligned else None,
        'device':str(device),'runtime_seconds':time.time()-start,'test_evaluated':args.evaluate_test,
        'python':platform.python_version(),'torch':str(torch.__version__),
        'split_file':str(args.split.resolve()),'split_sha256':sha256(args.split),
        'normalization_sha256':sha256(args.normalization),
        'sizes':{k:len(v) for k,v in indices.items()},
        'limitations':['Single strict split and one architecture configuration.',
          'Training optimizes MSE although primary model selection metric is Tau-c.',
          'Input CQT is approximately 5 fps; fine onset timing is unavailable.',
          'Convolution BatchNorm/padding behavior is unchanged; GRU and attention mask only.',
          'When rhythm is used: nominal 5 fps and common origin, not waveform-verified alignment.',
          'Rhythm tempo_confidence is a tempogram peak/mean ratio, not calibrated BPM confidence.',
          'Invalid rhythm values use train-mean imputation without extra indicator channels.',
          'No test evaluation unless explicitly requested.']}
    write_json(args.output/'config.json',config); write_json(args.output/'metrics.json',results)
    write_json(args.output/'history.json',history)
    for name,rows in predictions.items(): write_json(args.output/f'predictions_{name}.json',rows)
    torch.save({'state_dict':best['state'],'config':config,'normalization_sha256':sha256(args.normalization)},args.output/'model.pt')
    print(json.dumps({'output':str(args.output),'best_epoch':best['epoch'],'results':results},ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cqt-dir',type=Path,default=ROOT/'datasets/features/cqt')
    p.add_argument('--metadata',type=Path,default=ROOT/'datasets/metadata/new_clean_data.json')
    p.add_argument('--split',type=Path,default=ROOT/'datasets/splits/cqt_composer_split.json')
    p.add_argument('--normalization',type=Path,default=ROOT/'datasets/features/cqt_train_normalization.npz')
    p.add_argument('--feature-mode',choices=['cqt','cqt-rhythm'],default='cqt')
    p.add_argument('--rhythm-dir',type=Path,default=None,
                   help='Also supply for CQT-only control to use identical common-time cropping')
    p.add_argument('--manifest',type=Path,default=None,help='Defaults to the manifest recorded in the split')
    p.add_argument('--cqt-fps',type=float,default=5.0)
    p.add_argument('--cqt-time-offset',type=float,default=0.0)
    p.add_argument('--output',type=Path,default=ROOT/'outputs/cqt_crnn/run_01')
    p.add_argument('--device',choices=['auto','cpu','mps','cuda'],default='auto')
    p.add_argument('--seed',type=int,default=42); p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--num-workers',type=int,default=4); p.add_argument('--prefetch-factor',type=int,default=2)
    p.add_argument('--pin-memory',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--amp',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--conv-channels',type=int,default=64); p.add_argument('--hidden-size',type=int,default=64)
    p.add_argument('--dropout',type=float,default=0.2); p.add_argument('--learning-rate',type=float,default=1e-3)
    p.add_argument('--weight-decay',type=float,default=1e-4); p.add_argument('--grad-clip',type=float,default=1.0)
    p.add_argument('--max-epochs',type=int,default=100,help='Maximum epochs (default: 100)')
    p.add_argument('--patience',type=int,default=15,help='Epochs without meaningful Tau-c gain (default: 15)')
    p.add_argument('--min-delta',type=float,default=0.001,
                   help='Minimum cumulative Tau-c gain to reset patience (default: 0.001)')
    p.add_argument('--evaluate-test',action='store_true')
    main(p.parse_args())
