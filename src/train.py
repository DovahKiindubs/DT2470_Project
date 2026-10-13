"""CQT-only Ridge 基线。默认仅验证集选参，不读取测试集预测表现。"""
import argparse
import json
import platform
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate import evaluate
from data.feature_matrix import prepare_features
from features_cqt import ROOT, sha256
from data.split_data import make_groups, make_split, validate_split


def write_json(path, data):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def build_model(alpha):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver='svd'))


def _feature_group_name(args):
    name = args.features.name.lower()
    if 'structure' in name:
        base = 'structure'
    elif 'rhythm' in name:
        base = 'rhythm'
    elif 'hard_windows' in name:
        base = 'hard_windows'
    else:
        base = 'cqt'
    parts = [base]
    if args.rhythm_features and base != 'rhythm':
        parts.append('rhythm')
    if args.structure_features and base != 'structure':
        parts.append('structure')
    return '+'.join(parts)


def run(args):
    if args.output.exists():
        raise FileExistsError(f'Choose a new run directory; refusing overwrite: {args.output}')
    if not args.alphas or any(not np.isfinite(a) or a <= 0 for a in args.alphas):
        raise ValueError('All Ridge alphas must be finite and positive')
    metadata = json.loads(args.metadata.read_text(encoding='utf-8'))
    current_hash = sha256(args.metadata)
    if args.split.exists():
        split_doc = json.loads(args.split.read_text(encoding='utf-8'))
        parts = split_doc['parts']
        keys = sorted(set(parts['train']) | set(parts['val']) | set(parts['test']))
    else:
        extras = [p for p in (args.rhythm_features, args.structure_features) if p]
        if extras:
            raise ValueError('Pass --split when using extra feature groups')
        with np.load(args.features, allow_pickle=False) as data:
            keys = sorted(map(str, data['ids']))
        parts = split_doc = None
    extras = [p for p in (args.rhythm_features, args.structure_features) if p]
    X, y, ids, feature_names, cache_hash = prepare_features(
        args.features, metadata, keys, extras)
    # 标签已对齐当前 metadata；split 用其记录的 hash（全量旧 split 或子集新 hash）。
    if parts is None:
        if cache_hash != current_hash:
            raise ValueError('Metadata changed since feature extraction; rebuild the cache')
        metadata_hash = cache_hash
        groups = make_groups(ids, metadata)
        parts = make_split(ids, groups, args.seed)
        split_doc = {'seed': args.seed, 'metadata_sha256': metadata_hash,
                     'method': 'GroupShuffleSplit: 20% test groups, then 25% validation of remaining groups',
                     'grouping': 'Connected components of normalized composer strings and shared YouTube IDs',
                     'note': 'Approximate 60/20/20 by groups, not exact record counts; no grade stratification',
                     'parts': parts}
    else:
        metadata_hash = split_doc['metadata_sha256']
        if metadata_hash not in (cache_hash, current_hash):
            raise ValueError('Metadata differs from frozen split')
    if not set(map(str, ids)) <= set(metadata):
        raise ValueError('Feature/label/metadata alignment failed')
    groups = make_groups(ids, metadata)
    indices = validate_split(parts, ids, groups)
    # 必须先确认覆盖；不为得到更好的成绩反复更换 seed。
    coverage = {name: {str(level): int(np.sum(y[ix] == level)) for level in range(11)}
                for name, ix in indices.items()}
    if any(count == 0 for counts in coverage.values() for count in counts.values()):
        raise ValueError('At least one split lacks a grade. Review split design before training.')
    if not args.split.exists():
        args.split.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.split, split_doc)
    train, val = indices['train'], indices['val']
    trials, models = [], []
    for alpha in sorted(set(args.alphas)):
        model = build_model(alpha)
        model.fit(X[train], y[train])
        metrics = evaluate(y[val], model.predict(X[val]))
        trials.append({'alpha': alpha, 'validation': metrics})
        models.append(model)
    def score(i):
        metrics = trials[i]['validation']
        tau = metrics['kendall_tau_c']
        return (-np.inf if tau is None else tau, -metrics['mse'])
    winner = max(range(len(trials)), key=score)
    if trials[winner]['validation']['kendall_tau_c'] is None:
        raise ValueError('No model has a defined validation rank correlation')
    model = models[winner]
    mean_grade = float(y[train].mean())
    results = {'validation': trials[winner]['validation'],
               'mean_baseline_validation': evaluate(y[val], np.full(len(val), mean_grade)),
               'test_evaluated': args.evaluate_test}
    predictions = {}
    for name in (('val', 'test') if args.evaluate_test else ('val',)):
        ix = indices[name]
        pred = model.predict(X[ix])
        predictions[name] = [{'id': str(ids[i]), 'true_grade': int(y[i]), 'score': float(p)}
                             for i, p in zip(ix, pred)]
        if name == 'test':
            results['test'] = evaluate(y[ix], pred)
            results['mean_baseline_test'] = evaluate(y[ix], np.full(len(ix), mean_grade))
    config = {
        'feature_group': _feature_group_name(args),
        'model': 'StandardScaler + Ridge(solver=svd)',
        'chosen_alpha': trials[winner]['alpha'], 'selection': 'validation Tau-c, then lower validation MSE',
        'alphas': sorted(set(args.alphas)), 'python': platform.python_version(),
        'versions': {p: version(p) for p in ('numpy', 'scipy', 'scikit-learn', 'joblib')},
        'features_file': str(args.features.resolve()), 'features_sha256': sha256(args.features),
        'rhythm_features_file': str(args.rhythm_features.resolve()) if args.rhythm_features else None,
        'structure_features_file': str(args.structure_features.resolve()) if args.structure_features else None,
        'split_file': str(args.split.resolve()), 'split_sha256': sha256(args.split),
        'split_seed': split_doc['seed'], 'metadata_sha256': metadata_hash,
        'sizes': {name: len(ix) for name, ix in indices.items()}, 'grade_counts': coverage,
        'n_features': len(feature_names),
        'preprocessing': f'{len(feature_names)} cached features; scaler fitted on train only',
        'metric_conventions': 'Raw Ridge scores for Tau-c/Spearman/MAE/MSE; Acc±1 clips to 0..10 then floor(x+0.5)',
        'limitations': ['Single fixed split; not official MIREX evaluation.',
                       'All available CQT records; later rhythm comparison must use same common audio subset.',
                       'Composer strings normalized for Unicode/case/whitespace, not musicological aliases.',
                       'Ridge/MSE treat integer grades numerically; this is a baseline, not interval-free modeling.'],
        'code_sha256': {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
    }
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output/'config.json', config)
    write_json(args.output/'metrics.json', results)
    write_json(args.output/'validation_search.json', trials)
    for name, rows in predictions.items():
        write_json(args.output/f'predictions_{name}.json', rows)
    joblib.dump({'pipeline': model, 'feature_names': feature_names, 'config': config}, args.output/'model.joblib')
    print(json.dumps({'output': str(args.output), 'alpha': config['chosen_alpha'],
                      'sizes': config['sizes'], 'results': results}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, default=ROOT/'datasets/features/cqt_stats.npz')
    parser.add_argument('--rhythm-features', type=Path, default=None)
    parser.add_argument('--structure-features', type=Path, default=None)
    parser.add_argument('--metadata', type=Path, default=ROOT/'datasets/metadata/new_clean_data.json')
    parser.add_argument('--split', type=Path, default=ROOT/'datasets/splits/cqt_composer_split.json')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/cqt_only/ridge/run_01')
    parser.add_argument('--seed', type=int, default=42, help='Used only when first creating the frozen split')
    parser.add_argument('--alphas', type=float, nargs='+', default=[0.1, 1, 10, 100, 1000])
    parser.add_argument('--evaluate-test', action='store_true', help='Explicit final holdout evaluation after freezing the method')
    run(parser.parse_args())
