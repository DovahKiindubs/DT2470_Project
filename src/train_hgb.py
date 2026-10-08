"""CQT-only Histogram Gradient Boosting 对照；默认仅验证集选参。"""
import argparse
import json
import platform
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from evaluate import evaluate
from features_cqt import ROOT, sha256
from split_data import make_groups, validate_split


def write_json(path, data):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def build_hgb(max_leaf_nodes, max_iter=300, learning_rate=0.05, random_state=42):
    return HistGradientBoostingRegressor(
        loss='squared_error', learning_rate=learning_rate, max_iter=max_iter,
        max_leaf_nodes=max_leaf_nodes, min_samples_leaf=20, l2_regularization=1.0,
        early_stopping=False, random_state=random_state,
    )


def run(args):
    if args.output.exists():
        raise FileExistsError(f'Choose a new run directory; refusing overwrite: {args.output}')
    leaves = sorted(set(args.leaf_nodes))
    if not leaves or any(value < 2 for value in leaves) or args.max_iter <= 0 or args.learning_rate <= 0:
        raise ValueError('Invalid tree parameters')
    if not args.split.exists():
        raise FileNotFoundError('Reuse the frozen Ridge split; run the baseline first')
    metadata = json.loads(args.metadata.read_text(encoding='utf-8'))
    with np.load(args.features, allow_pickle=False) as data:
        X, y, ids = data['X'], data['y'], data['ids']
        feature_names = data['feature_names'].tolist()
        metadata_hash = str(data['metadata_sha256'].item())
    if metadata_hash != sha256(args.metadata):
        raise ValueError('Metadata changed since feature extraction')
    if X.ndim != 2 or X.shape != (len(ids), len(feature_names)) or y.shape != (len(ids),) or not np.isfinite(X).all():
        raise ValueError('Unexpected or non-finite feature cache')
    if not feature_names or len(set(ids)) != len(ids):
        raise ValueError('Missing feature names or duplicate ids')
    if set(ids) != set(metadata) or not np.array_equal(y, [int(metadata[k]['ps']) for k in ids]):
        raise ValueError('Feature/label/metadata alignment failed')
    split_doc = json.loads(args.split.read_text(encoding='utf-8'))
    if split_doc['metadata_sha256'] != metadata_hash:
        raise ValueError('Metadata differs from frozen split')
    indices = validate_split(split_doc['parts'], ids, make_groups(ids, metadata))
    train, val = indices['train'], indices['val']
    trials, models = [], []
    for nodes in leaves:
        model = build_hgb(nodes, args.max_iter, args.learning_rate, args.seed)
        model.fit(X[train], y[train])
        metrics = evaluate(y[val], model.predict(X[val]))
        trials.append({'max_leaf_nodes': nodes, 'validation': metrics})
        models.append(model)
    def score(index):
        metrics = trials[index]['validation']
        tau = metrics['kendall_tau_c']
        return (-np.inf if tau is None else tau, -metrics['mse'])
    winner = max(range(len(trials)), key=score)
    model = models[winner]
    if trials[winner]['validation']['kendall_tau_c'] is None:
        raise ValueError('No model has a defined validation rank correlation')
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
        'feature_group': 'cqt_only',
        'model': 'HistGradientBoostingRegressor',
        'chosen_max_leaf_nodes': trials[winner]['max_leaf_nodes'],
        'selection': 'validation Tau-c, then lower validation MSE',
        'max_leaf_nodes_candidates': leaves,
        'fixed_parameters': {'loss': 'squared_error', 'learning_rate': args.learning_rate,
            'max_iter': args.max_iter, 'min_samples_leaf': 20, 'l2_regularization': 1.0,
            'early_stopping': False, 'random_state': args.seed},
        'python': platform.python_version(),
        'versions': {p: version(p) for p in ('numpy', 'scipy', 'scikit-learn', 'joblib')},
        'features_file': str(args.features.resolve()), 'features_sha256': sha256(args.features),
        'split_file': str(args.split.resolve()), 'split_sha256': sha256(args.split),
        'split_seed': split_doc['seed'], 'model_seed': args.seed,
        'metadata_sha256': metadata_hash,
        'sizes': {name: len(ix) for name, ix in indices.items()},
        'n_features': len(feature_names),
        'preprocessing': f'{len(feature_names)} cached CQT features; no StandardScaler for trees',
        'metric_conventions': 'Raw model scores for Tau-c/Spearman/MAE/MSE; Acc±1 clips to 0..10 then floor(x+0.5)',
        'limitations': ['Single fixed split; not official MIREX evaluation.',
                        'Validation set used to select one of five tree-complexity candidates.',
                        'Temporal order already removed by mean/std pooling.',
                        'Later rhythm comparison must use the same common audio subset.'],
        'code_sha256': {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
    }
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output/'config.json', config)
    write_json(args.output/'metrics.json', results)
    write_json(args.output/'validation_search.json', trials)
    for name, rows in predictions.items():
        write_json(args.output/f'predictions_{name}.json', rows)
    joblib.dump({'model': model, 'feature_names': feature_names, 'config': config},
                args.output/'model.joblib')
    print(json.dumps({'output': str(args.output),
                      'max_leaf_nodes': config['chosen_max_leaf_nodes'],
                      'sizes': config['sizes'], 'results': results}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, default=ROOT/'datasets/features/cqt_stats.npz')
    parser.add_argument('--metadata', type=Path, default=ROOT/'datasets/metadata/new_clean_data.json')
    parser.add_argument('--split', type=Path, default=ROOT/'datasets/splits/cqt_composer_split.json')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/cqt_only/hgb/run_01')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--leaf-nodes', type=int, nargs='+', default=[7, 15, 31, 63, 127])
    parser.add_argument('--max-iter', type=int, default=300)
    parser.add_argument('--learning-rate', type=float, default=0.05)
    parser.add_argument('--evaluate-test', action='store_true')
    run(parser.parse_args())
