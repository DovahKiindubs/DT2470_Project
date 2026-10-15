"""将可信 PSyllabus CQT 转为逐曲均值/标准差；不修改原始 .bin。"""
import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


class ArrayUnpickler(pickle.Unpickler):
    """只允许官方文件使用的 NumPy 数组类型；仍只处理可信来源文件。"""
    def find_class(self, module, name):
        if module == 'numpy' and name in ('ndarray', 'dtype'):
            return getattr(np, name)
        if module in ('numpy.core.multiarray', 'numpy._core.multiarray') and name == '_reconstruct':
            from numpy._core.multiarray import _reconstruct
            return _reconstruct
        raise pickle.UnpicklingError(f'Blocked pickle global: {module}.{name}')


def summarize_cqt(cqt):
    if not isinstance(cqt, np.ndarray) or cqt.ndim != 2:
        raise ValueError('CQT must be a 2-D NumPy array')
    if cqt.shape[0] != 88 or cqt.shape[1] == 0:
        raise ValueError(f'Expected (88, T>0), got {cqt.shape}')
    if cqt.dtype.kind != 'f' or not np.isfinite(cqt).all():
        raise ValueError('CQT must contain finite real floating-point values')
    # 输入已是 dB，不再取 log/abs，不做逐曲 z-score。
    return np.concatenate((cqt.mean(axis=1, dtype=np.float64),
                           cqt.std(axis=1, dtype=np.float64, ddof=0)))


def extract(cqt_dir, metadata_path, output):
    cqt_dir, metadata_path, output = map(Path, (cqt_dir, metadata_path, output))
    if output.exists():
        raise FileExistsError(f'Feature cache already exists: {output}')
    if output.suffix != '.npz':
        raise ValueError('Feature output must end in .npz')
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    if not isinstance(metadata, dict) or not metadata:
        raise ValueError('Metadata must be a nonempty object')
    files = {p.stem: p for p in cqt_dir.glob('*.bin')}
    if set(files) != set(metadata):
        raise ValueError(f'CQT/metadata mismatch: {len(set(metadata)-set(files))} missing, '
                         f'{len(set(files)-set(metadata))} extra; no silent exclusions')
    ids, features, labels, source_hashes = sorted(metadata), [], [], []
    for i, key in enumerate(ids, 1):
        raw_grade = metadata[key].get('ps')
        if str(raw_grade) not in {str(n) for n in range(11)}:
            raise ValueError(f'Invalid grade for {key}: {raw_grade}')
        path = files[key]
        if path.is_symlink() or path.stat().st_size > 64 * 1024**2:
            raise ValueError(f'Refusing symlink or unexpectedly large CQT: {path}')
        with path.open('rb') as f:
            cqt = ArrayUnpickler(f).load()
        features.append(summarize_cqt(cqt))
        labels.append(int(raw_grade))
        source_hashes.append(sha256(path))
        if i % 1000 == 0:
            print(f'Extracted {i}/{len(ids)}', flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    names = [f'{stat}_bin_{j:02d}' for stat in ('mean', 'std') for j in range(88)]
    with output.open('xb') as f:
        np.savez_compressed(f, X=np.asarray(features), y=np.asarray(labels, dtype=np.int64),
                            ids=np.asarray(ids), feature_names=np.asarray(names),
                            cqt_sha256=np.asarray(source_hashes),
                            metadata_sha256=np.asarray(sha256(metadata_path)))
    print(f'Saved {len(ids)} recordings x {len(names)} features: {output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cqt-dir', type=Path, default=ROOT/'datasets/features/cqt')
    parser.add_argument('--metadata', type=Path, default=ROOT/'datasets/metadata/new_clean_data.json')
    parser.add_argument('--output', type=Path, default=ROOT/'datasets/features/cqt_stats.npz')
    args = parser.parse_args()
    extract(args.cqt_dir, args.metadata, args.output)
