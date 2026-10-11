"""按 split ids 选取特征，可选与 rhythm 内存拼接。"""
from pathlib import Path

import numpy as np


def _load(path):
    with np.load(path, allow_pickle=False) as data:
        return (np.asarray(data['X']), np.asarray(data['y']), np.asarray(data['ids']),
                [str(n) for n in data['feature_names'].tolist()],
                str(data['metadata_sha256'].item()))


def _gather(X, y, ids, names, keys):
    lookup = {str(k): i for i, k in enumerate(ids)}
    missing = [k for k in keys if k not in lookup]
    if missing:
        raise ValueError(f'{len(missing)} keys missing from feature cache')
    ix = np.asarray([lookup[k] for k in keys])
    return X[ix], y[ix], np.asarray(keys), names


def prepare_features(features_path, metadata, keys, rhythm_path=None):
    """返回 X, y, ids, feature_names, cache_metadata_sha256。"""
    X, y, ids, names, cache_hash = _load(features_path)
    if X.ndim != 2 or X.shape != (len(ids), len(names)) or y.shape != (len(ids),):
        raise ValueError('Unexpected feature cache shape')
    if not np.isfinite(X).all() or len(set(map(str, ids))) != len(ids) or not names:
        raise ValueError('Invalid features, duplicate ids or feature names')
    X, y, ids, names = _gather(X, y, ids, names, keys)
    expected = np.asarray([int(metadata[k]['ps']) for k in keys], dtype=np.int64)
    if not np.array_equal(y, expected):
        raise ValueError('Feature labels disagree with metadata')
    if rhythm_path is None:
        return X, expected, ids, names, cache_hash
    rX, ry, rids, rnames, rhash = _load(rhythm_path)
    rX, ry, _, rnames = _gather(rX, ry, rids, rnames, keys)
    if not np.array_equal(ry, expected):
        raise ValueError('Rhythm labels disagree with metadata')
    return np.hstack([X, rX]), expected, ids, names + rnames, rhash
