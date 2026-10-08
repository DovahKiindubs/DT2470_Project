"""从 5 fps CQT 提取重叠滑窗与困难片段池化特征。"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from features_cqt import ArrayUnpickler, ROOT, sha256, summarize_cqt


def window_starts(n_frames, window_frames, hop_frames):
    if n_frames <= 0 or window_frames <= 0 or hop_frames <= 0:
        raise ValueError('Frame and window sizes must be positive')
    if n_frames <= window_frames:
        return [0]
    starts = list(range(0, n_frames-window_frames+1, hop_frames))
    last = n_frames-window_frames
    if starts[-1] != last:
        starts.append(last)
    return starts


def spectral_flux_score(window):
    """标签无关的局部变化代理：CQT dB 相邻帧正向变化的均值。"""
    if window.shape[1] < 2:
        return 0.0
    return float(np.maximum(np.diff(window, axis=1), 0).mean(dtype=np.float64))


def summarize_hard_windows(cqt, window_frames=50, hop_frames=25, hard_fraction=0.2):
    # 先复用全局校验；输入已是 dB，不再 log/abs/逐曲 z-score。
    global_summary = summarize_cqt(cqt)
    if not 0 < hard_fraction <= 1:
        raise ValueError('hard_fraction must be in (0, 1]')
    summaries, scores = [], []
    for start in window_starts(cqt.shape[1], window_frames, hop_frames):
        window = cqt[:, start:min(start+window_frames, cqt.shape[1])]
        summaries.append(summarize_cqt(window))
        scores.append(spectral_flux_score(window))
    summaries, scores = np.asarray(summaries), np.asarray(scores)
    n_hard = max(1, math.ceil(len(scores)*hard_fraction))
    # 稳定排序使分数相同时结果可复现；选择最高分窗口。
    hard_index = np.argsort(scores, kind='stable')[-n_hard:]
    hard = summaries[hard_index]
    features = np.concatenate((global_summary, hard.mean(axis=0), hard.max(axis=0)))
    if features.shape != (528,) or not np.isfinite(features).all():
        raise ValueError('Invalid pooled feature vector')
    diagnostics = {'n_frames': int(cqt.shape[1]), 'n_windows': int(len(scores)),
                   'n_hard_windows': int(n_hard), 'max_flux': float(scores.max()),
                   'mean_flux': float(scores.mean())}
    return features, diagnostics


def extract(cqt_dir, metadata_path, output, window_frames=50, hop_frames=25, hard_fraction=0.2):
    cqt_dir, metadata_path, output = map(Path, (cqt_dir, metadata_path, output))
    if output.exists():
        raise FileExistsError(f'Feature cache already exists: {output}')
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    files = {p.stem: p for p in cqt_dir.glob('*.bin')}
    if set(files) != set(metadata):
        raise ValueError('CQT/metadata mismatch; no silent exclusions')
    ids, features, labels, diagnostics, source_hashes = sorted(metadata), [], [], [], []
    for i, key in enumerate(ids, 1):
        raw_grade = metadata[key].get('ps')
        if str(raw_grade) not in {str(n) for n in range(11)}:
            raise ValueError(f'Invalid grade for {key}: {raw_grade}')
        path = files[key]
        if path.is_symlink() or path.stat().st_size > 64*1024**2:
            raise ValueError(f'Refusing symlink or unexpectedly large CQT: {path}')
        with path.open('rb') as f:
            cqt = ArrayUnpickler(f).load()
        vector, info = summarize_hard_windows(cqt, window_frames, hop_frames, hard_fraction)
        features.append(vector)
        labels.append(int(raw_grade))
        diagnostics.append([info[k] for k in ('n_frames','n_windows','n_hard_windows','max_flux','mean_flux')])
        source_hashes.append(sha256(path))
        if i % 1000 == 0:
            print(f'Extracted {i}/{len(ids)}', flush=True)
    base = [f'{stat}_bin_{j:02d}' for stat in ('mean','std') for j in range(88)]
    names = ([f'global_{name}' for name in base] + [f'hard_mean_{name}' for name in base]
             + [f'hard_max_{name}' for name in base])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as f:
        np.savez_compressed(f, X=np.asarray(features), y=np.asarray(labels,dtype=np.int64),
            ids=np.asarray(ids), feature_names=np.asarray(names), cqt_sha256=np.asarray(source_hashes),
            metadata_sha256=np.asarray(sha256(metadata_path)),
            extraction=np.asarray(json.dumps({'frame_rate_assumption_hz':5,'window_frames':window_frames,
                'hop_frames':hop_frames,'hard_fraction':hard_fraction,
                'difficulty_proxy':'mean positive adjacent-frame CQT-dB difference',
                'pooling':'global summary + hard-window summary mean + elementwise max'})),
            diagnostics=np.asarray(diagnostics,dtype=np.float64),
            diagnostic_names=np.asarray(['n_frames','n_windows','n_hard_windows','max_flux','mean_flux']))
    print(f'Saved {len(ids)} recordings x {len(names)} features: {output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cqt-dir',type=Path,default=ROOT/'datasets/features/cqt')
    parser.add_argument('--metadata',type=Path,default=ROOT/'datasets/metadata/new_clean_data.json')
    parser.add_argument('--output',type=Path,default=ROOT/'datasets/features/cqt_hard_windows.npz')
    parser.add_argument('--window-frames',type=int,default=50)
    parser.add_argument('--hop-frames',type=int,default=25)
    parser.add_argument('--hard-fraction',type=float,default=0.2)
    args=parser.parse_args()
    extract(args.cqt_dir,args.metadata,args.output,args.window_frames,args.hop_frames,args.hard_fraction)
