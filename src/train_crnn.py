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

from evaluate import evaluate
from features_cqt import ROOT, sha256
from model_crnn import CQTCRNN, CQTSequenceDataset, LengthBucketBatchSampler, collate_sequences
from split_data import make_groups, validate_split


def write_json(path,data):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False); f.write('\n')


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
    seed_all(args.seed)
    metadata=json.loads(args.metadata.read_text(encoding='utf-8'))
    split_doc=json.loads(args.split.read_text(encoding='utf-8'))
    if split_doc['metadata_sha256'] != sha256(args.metadata): raise ValueError('Split/metadata mismatch')
    ids=np.asarray(sorted(metadata)); labels=np.asarray([int(metadata[k]['ps']) for k in ids])
    indices=validate_split(split_doc['parts'],ids,make_groups(ids,metadata))
    with np.load(args.normalization,allow_pickle=False) as norm:
        mean,std=norm['mean'],norm['std']
        if str(norm['split_sha256'].item()) != sha256(args.split):
            raise ValueError('Normalization was not computed from this split')
    datasets={name:CQTSequenceDataset(ids[ix],labels[ix],args.cqt_dir,mean,std)
              for name,ix in indices.items()}
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
    model=CQTCRNN(args.conv_channels,args.hidden_size,args.dropout).to(device)
    amp_enabled=args.amp and device.type=='cuda'
    scaler=torch.amp.GradScaler('cuda',enabled=amp_enabled)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)
    loss_fn=nn.MSELoss()
    args.output.mkdir(parents=True,exist_ok=False)
    best_path=args.output/'best_checkpoint.pt'
    best=None; stale=0; history=[]; start=time.time()
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
        row={'epoch':epoch,'train_mse':total/count,'validation':metrics}
        history.append(row); print(json.dumps(row,ensure_ascii=False),flush=True)
        key=(-np.inf if tau is None else tau,-metrics['mse'])
        if best is None or key>best['key']:
            best={'key':key,'epoch':epoch,'state':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'metrics':metrics}; stale=0
            torch.save({'state_dict':best['state'],'epoch':epoch,'metrics':metrics,
                        'normalization_sha256':sha256(args.normalization)},best_path)
        else: stale += 1
        if stale>=args.patience: break
    model.load_state_dict(best['state']); model.to(device)
    predictions={}; results={'validation':best['metrics'],'test_evaluated':args.evaluate_test}
    for name in (('val','test') if args.evaluate_test else ('val',)):
        batch_ids,y,pred=predict(model,loaders[name],device,amp_enabled)
        predictions[name]=[{'id':key,'true_grade':int(t),'score':float(p)} for key,t,p in zip(batch_ids,y,pred)]
        if name=='test': results['test']=evaluate(y,pred)
    config={'feature_group':'cqt_sequence','model':'Conv1d x2 + BiGRU + masked attention + linear head',
        'architecture':{'input_bins':88,'conv_channels':args.conv_channels,'conv_kernel':5,'conv_time_stride':4,
          'gru_hidden_size_per_direction':args.hidden_size,'bidirectional':True,'dropout':args.dropout},
        'training':{'loss':'MSE','optimizer':'AdamW','learning_rate':args.learning_rate,
          'weight_decay':args.weight_decay,'batch_size':args.batch_size,'max_epochs':args.max_epochs,
          'patience':args.patience,'gradient_clip':args.grad_clip,'seed':args.seed,
          'num_workers':args.num_workers,'pin_memory':args.pin_memory,'amp':amp_enabled,
          'selection':'validation Tau-c, then lower validation MSE','best_epoch':best['epoch']},
        'normalization':'Per-CQT-bin mean/std computed across all train frames only',
        'device':str(device),'runtime_seconds':time.time()-start,'test_evaluated':args.evaluate_test,
        'python':platform.python_version(),'torch':str(torch.__version__),
        'split_file':str(args.split.resolve()),'split_sha256':sha256(args.split),
        'normalization_sha256':sha256(args.normalization),
        'sizes':{k:len(v) for k,v in indices.items()},
        'limitations':['Single strict split and one architecture configuration.',
          'Training optimizes MSE although primary model selection metric is Tau-c.',
          'Input CQT is approximately 5 fps; fine onset timing is unavailable.',
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
    p.add_argument('--output',type=Path,default=ROOT/'outputs/cqt_crnn/run_01')
    p.add_argument('--device',choices=['auto','cpu','mps','cuda'],default='auto')
    p.add_argument('--seed',type=int,default=42); p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--num-workers',type=int,default=4); p.add_argument('--prefetch-factor',type=int,default=2)
    p.add_argument('--pin-memory',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--amp',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--conv-channels',type=int,default=64); p.add_argument('--hidden-size',type=int,default=64)
    p.add_argument('--dropout',type=float,default=0.2); p.add_argument('--learning-rate',type=float,default=1e-3)
    p.add_argument('--weight-decay',type=float,default=1e-4); p.add_argument('--grad-clip',type=float,default=1.0)
    p.add_argument('--max-epochs',type=int,default=30); p.add_argument('--patience',type=int,default=5)
    p.add_argument('--evaluate-test',action='store_true')
    main(p.parse_args())
