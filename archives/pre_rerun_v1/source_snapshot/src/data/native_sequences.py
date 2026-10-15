"""Native-resolution CQT/rhythm sequences for two-branch late fusion."""
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from data.aligned_sequences import (RHYTHM_NAMES, check_metadata_hash, digest, nfc,
                                    scan_files)
from features_cqt import ArrayUnpickler, ROOT, sha256


def rhythm_validity(rhythm):
    rhythm = np.asarray(rhythm)
    if rhythm.ndim != 2 or rhythm.shape[0] != 9 or rhythm.shape[1] == 0:
        raise ValueError('Rhythm must be (9,N>0)')
    if not np.isfinite(rhythm).all() or np.any(rhythm < 0):
        raise ValueError('Rhythm values must be finite and nonnegative')
    valid = np.ones(rhythm.shape, dtype=bool)
    valid[4:7] = rhythm[4] > 0
    active = (rhythm[0] > 0) & (rhythm[1] > 0)
    valid[7] = active & (rhythm[4] > 0)
    valid[8] = active
    return valid


def _safe_cqt(data):
    cqt = ArrayUnpickler(io.BytesIO(data)).load()
    if (not isinstance(cqt, np.ndarray) or cqt.ndim != 2 or cqt.shape[0] != 88
            or cqt.shape[1] == 0 or cqt.dtype.kind != 'f' or not np.isfinite(cqt).all()):
        raise ValueError('CQT must be finite float (88,T>0)')
    return cqt


class NativePairSource:
    """Read immutable native CQT and rhythm sequences without temporal alignment."""
    def __init__(self, ids, labels, cqt_dir, rhythm_dir, metadata_path,
                 split_path, manifest_path=None):
        self.cqt = scan_files(cqt_dir, '.bin')
        self.rhythm = scan_files(rhythm_dir, '.npz')
        self.labels = {nfc(k): int(y) for k, y in zip(ids, labels)}
        if len(self.labels) != len(ids):
            raise ValueError('Duplicate sequence keys')
        missing = set(self.labels) - (set(self.cqt) & set(self.rhythm))
        if missing:
            raise ValueError(f'Missing paired files ({len(missing)}), e.g. {sorted(missing)[:3]}')
        index_path = Path(rhythm_dir)/'index.npz'
        with np.load(index_path, allow_pickle=False) as idx:
            index_ids = [nfc(k) for k in idx['ids']]
            if len(index_ids) != len(set(index_ids)):
                raise ValueError('Duplicate rhythm index keys')
            if idx['feature_names'].tolist() != list(RHYTHM_NAMES):
                raise ValueError('Unexpected rhythm channel order')
            check_metadata_hash(metadata_path, str(idx['metadata_sha256'].item()))
            self.win = float(idx['win_sec']); self.hop = float(idx['hop_sec'])
            if not np.isfinite([self.win,self.hop]).all() or min(self.win,self.hop) <= 0:
                raise ValueError('Invalid rhythm window settings')
            self.index = {key:(int(idx['y'][i]),int(idx['lengths'][i]),str(idx['audio_sha256'][i]))
                          for i,key in enumerate(index_ids)}
        if not set(self.labels) <= set(self.index):
            raise ValueError('Frozen keys missing from rhythm index')
        doc = json.loads(Path(split_path).read_text(encoding='utf-8'))
        if manifest_path is None and doc.get('manifest'):
            manifest_path = ROOT/doc['manifest']
        self.expected = {}
        self.manifest_sha256 = ''
        if manifest_path is not None:
            manifest_path = Path(manifest_path)
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            if manifest['split_sha256'] != sha256(split_path):
                raise ValueError('Manifest does not match frozen split')
            if manifest['source_sha256']['datasets/features/rhythm_windows/index.npz'] != sha256(index_path):
                raise ValueError('Rhythm index changed since freeze')
            records = {nfc(r['id']):r for r in manifest['records']}
            for key,label in self.labels.items():
                row = records.get(key)
                if row is None or row['status'] != 'included' or int(row['label']) != label:
                    raise ValueError(f'Not an included manifest member: {key}')
                self.expected[key] = (row['cqt_sha256'],row['rhythm_sha256'])
            self.manifest_sha256 = sha256(manifest_path)
        else:
            self.expected = {k:(sha256(self.cqt[k]),sha256(self.rhythm[k])) for k in self.labels}
        identity = {'pairs':sorted((k,*v) for k,v in self.expected.items()),
                    'index_sha256':sha256(index_path)}
        self.source_signature = digest(json.dumps(identity,ensure_ascii=False,sort_keys=True).encode())

    def read(self,key):
        key = nfc(key)
        cb,rb = self.cqt[key].read_bytes(),self.rhythm[key].read_bytes()
        if (digest(cb),digest(rb)) != self.expected[key]:
            raise ValueError(f'Feature file changed since freeze/preparation: {key}')
        cqt = _safe_cqt(cb)
        with np.load(io.BytesIO(rb),allow_pickle=False) as r:
            x,times = np.asarray(r['X']),np.asarray(r['times'],dtype=float)
            label,length,audio_hash = self.index[key]
            if (label != self.labels[key] or int(r['y']) != label or x.shape != (9,length)
                    or str(r['audio_sha256'].item()) != audio_hash
                    or r['feature_names'].tolist() != list(RHYTHM_NAMES)
                    or float(r['win_sec']) != self.win or float(r['hop_sec']) != self.hop):
                raise ValueError(f'Rhythm file/index mismatch: {key}')
            if (times.shape != (length,) or times[0] < 0 or not np.isfinite(times).all()
                    or np.any(np.diff(times) <= 0)):
                raise ValueError(f'Invalid rhythm time axis: {key}')
            valid = rhythm_validity(x)
            return cqt,x,valid


def _merge_stats(counts,mean,m2,channel,values):
    values = np.asarray(values,dtype=np.float64)
    n = values.size
    if n == 0:
        return
    local = values.mean(); delta = local-mean[channel]; total = counts[channel]+n
    m2[channel] += np.square(values-local).sum()+delta**2*counts[channel]*n/total
    mean[channel] += delta*n/total;counts[channel] = total


def compute_native_statistics(source,train_ids):
    c_counts=np.zeros(88,dtype=np.int64);c_mean=np.zeros(88);c_m2=np.zeros(88)
    r_counts=np.zeros(9,dtype=np.int64);r_mean=np.zeros(9);r_m2=np.zeros(9)
    c_frames=r_windows=0
    for i,key in enumerate(train_ids,1):
        cqt,rhythm,valid=source.read(key);c_frames+=cqt.shape[1];r_windows+=rhythm.shape[1]
        for channel in range(88):_merge_stats(c_counts,c_mean,c_m2,channel,cqt[channel])
        for channel in range(9):_merge_stats(r_counts,r_mean,r_m2,channel,rhythm[channel,valid[channel]])
        if i%500==0:print(f'Native train statistics {i}/{len(train_ids)}',flush=True)
    if not c_frames or not r_windows or np.any(c_counts==0) or np.any(r_counts==0):
        raise ValueError('Empty/all-invalid native train channel')
    c_std=np.sqrt(c_m2/c_counts);r_std=np.sqrt(r_m2/r_counts)
    c_std[c_std<1e-8]=1.;r_std[r_std<1e-8]=1.
    return c_mean,c_std,c_counts,r_mean,r_std,r_counts,c_frames,r_windows


def load_native_normalization(path,source,split_path,metadata_path,train_ids):
    with np.load(path,allow_pickle=False) as d:
        required={'cqt_mean','cqt_std','cqt_counts','rhythm_mean','rhythm_std','rhythm_counts',
                  'source_signature','split_sha256','metadata_sha256','train_ids_sha256','mode'}
        if not required<=set(d.files):
            raise ValueError('Need native late-fusion normalization cache')
        if (str(d['mode'].item())!='native_two_branch_v1'
                or str(d['split_sha256'].item())!=sha256(split_path)
                or str(d['train_ids_sha256'].item())!=digest('\n'.join(train_ids).encode())
                or str(d['source_signature'].item())!=source.source_signature):
            raise ValueError('Native normalization provenance mismatch')
        check_metadata_hash(metadata_path,str(d['metadata_sha256'].item()))
        cm,cs,rm,rs=d['cqt_mean'],d['cqt_std'],d['rhythm_mean'],d['rhythm_std']
        if cm.shape!=(88,) or cs.shape!=(88,) or rm.shape!=(9,) or rs.shape!=(9,):
            raise ValueError('Unexpected native normalization shapes')
        if not all(np.isfinite(x).all() for x in (cm,cs,rm,rs)) or np.any(cs<=0) or np.any(rs<=0):
            raise ValueError('Invalid native normalization values')
        return cm.copy(),cs.copy(),rm.copy(),rs.copy()


class NativeTwoBranchDataset(Dataset):
    def __init__(self,ids,labels,source,cqt_mean,cqt_std,rhythm_mean,rhythm_std):
        self.ids=list(ids);self.labels=np.asarray(labels,dtype=np.float32);self.source=source
        self.cqt_mean=np.asarray(cqt_mean,dtype=np.float32).reshape(88,1)
        self.cqt_std=np.asarray(cqt_std,dtype=np.float32).reshape(88,1)
        self.rhythm_mean=np.asarray(rhythm_mean,dtype=np.float32).reshape(9,1)
        self.rhythm_std=np.asarray(rhythm_std,dtype=np.float32).reshape(9,1)
        self.cqt_lengths=[];self.rhythm_lengths=[]
        for key in self.ids:
            cqt,rhythm,_=source.read(key)
            self.cqt_lengths.append(cqt.shape[1]);self.rhythm_lengths.append(rhythm.shape[1])
        # CQT drives bucketing because it dominates padding and compute.
        self.lengths=self.cqt_lengths

    def __len__(self):return len(self.ids)

    def __getitem__(self,index):
        cqt,rhythm,valid=self.source.read(self.ids[index])
        cqt=(cqt.astype(np.float32)-self.cqt_mean)/self.cqt_std
        rhythm=(rhythm.astype(np.float32)-self.rhythm_mean)/self.rhythm_std
        rhythm[~valid]=0.
        if not np.isfinite(cqt).all() or not np.isfinite(rhythm).all():
            raise ValueError(f'Non-finite native sequence: {self.ids[index]}')
        return (torch.from_numpy(cqt.copy()),torch.from_numpy(rhythm.copy()),
                torch.tensor(self.labels[index]),self.ids[index])


def collate_native_pairs(batch):
    cqt,rhythm,labels,ids=zip(*batch)
    c_len=torch.tensor([x.shape[1] for x in cqt],dtype=torch.long)
    r_len=torch.tensor([x.shape[1] for x in rhythm],dtype=torch.long)
    c_pad=torch.zeros(len(batch),88,int(c_len.max()));r_pad=torch.zeros(len(batch),9,int(r_len.max()))
    for i,(c,r) in enumerate(zip(cqt,rhythm)):
        c_pad[i,:,:c.shape[1]]=c;r_pad[i,:,:r.shape[1]]=r
    return c_pad,c_len,r_pad,r_len,torch.stack(labels),list(ids)
