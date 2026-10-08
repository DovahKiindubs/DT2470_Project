"""训练集 CQT 统计：按全部有效帧计算每个频率 bin 的均值和标准差。"""
import argparse
import json
from pathlib import Path

import numpy as np

from features_cqt import ArrayUnpickler, ROOT, sha256


def compute(ids, cqt_dir):
    count = 0
    total = np.zeros(88, dtype=np.float64)
    total_sq = np.zeros(88, dtype=np.float64)
    for i, key in enumerate(ids, 1):
        path = Path(cqt_dir)/f'{key}.bin'
        with path.open('rb') as f:
            cqt = ArrayUnpickler(f).load()
        if cqt.ndim != 2 or cqt.shape[0] != 88 or cqt.shape[1] == 0 or not np.isfinite(cqt).all():
            raise ValueError(f'Invalid CQT: {path}')
        x = cqt.astype(np.float64, copy=False)
        total += x.sum(axis=1)
        total_sq += np.square(x).sum(axis=1)
        count += x.shape[1]
        if i % 1000 == 0: print(f'Scanned {i}/{len(ids)}', flush=True)
    mean = total/count
    variance = np.maximum(total_sq/count-np.square(mean), 1e-8)
    return mean, np.sqrt(variance), count


def main(args):
    if args.output.exists(): raise FileExistsError(args.output)
    split_doc = json.loads(args.split.read_text(encoding='utf-8'))
    metadata = json.loads(args.metadata.read_text(encoding='utf-8'))
    if split_doc['metadata_sha256'] != sha256(args.metadata):
        raise ValueError('Frozen split and metadata mismatch')
    ids = split_doc['parts']['train']
    if not ids or not set(ids) <= set(metadata): raise ValueError('Invalid train ids')
    mean, std, frames = compute(ids, args.cqt_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as f:
        np.savez(f, mean=mean, std=std, n_frames=np.asarray(frames),
                 train_ids_sha256=np.asarray(__import__('hashlib').sha256(
                     '\n'.join(ids).encode()).hexdigest()),
                 split_sha256=np.asarray(sha256(args.split)),
                 metadata_sha256=np.asarray(sha256(args.metadata)))
    print(f'Saved training-only normalization from {len(ids)} pieces, {frames} frames')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cqt-dir',type=Path,default=ROOT/'datasets/features/cqt')
    p.add_argument('--metadata',type=Path,default=ROOT/'datasets/metadata/new_clean_data.json')
    p.add_argument('--split',type=Path,default=ROOT/'datasets/splits/cqt_composer_split.json')
    p.add_argument('--output',type=Path,default=ROOT/'datasets/features/cqt_train_normalization.npz')
    main(p.parse_args())
