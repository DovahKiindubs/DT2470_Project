"""One experiment catalogue and safe server runner; never changes training algorithms."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
CATALOGUE = ROOT/'configs/rerun_v1.json'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value, exclusive=False):
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    if exclusive:
        with Path(path).open('x', encoding='utf-8') as stream:
            stream.write(text)
    else:
        temp = Path(path).with_name(Path(path).name+'.'+uuid4().hex+'.tmp')
        temp.write_text(text, encoding='utf-8')
        temp.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def select(catalogue, group='all', experiments=None):
    rows = catalogue['experiments']
    if experiments:
        unknown = set(experiments)-{r['id'] for r in rows}
        if unknown:
            raise ValueError(f'Unknown experiments: {sorted(unknown)}')
        rows = [r for r in rows if r['id'] in experiments]
    rows = [r for r in rows if group == 'all' or r['group'] == group]
    if not rows:
        raise ValueError('No experiments selected')
    return rows


def plan(catalogue, experiment, seed=42, run_name='run_01', device='cuda', workers=4, root=ROOT):
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}', run_name):
        raise ValueError('run-name must be a simple name, not a path')
    if not 0 <= seed < 2**32 or workers < 0:
        raise ValueError('Invalid seed/workers')
    group = catalogue['groups'][experiment['group']]
    run = root/'outputs'/catalogue['suite']/experiment['group']/experiment['folder']/f'seed_{seed}_{run_name}'
    args = list(group['args'])+list(experiment['args'])
    args += ['--metadata', catalogue['metadata'], '--split', catalogue['split'], '--seed', str(seed)]
    if experiment['group'] != 'hgb':
        args += ['--manifest', catalogue['manifest'], '--device', device, '--num-workers', str(workers)]
    args += ['--output', str(run/'artifacts')]
    # All invocations run from repository root. No shell evaluation of paths/arguments.
    return {'experiment_id': experiment['id'], 'group': experiment['group'],
            'run': run, 'command': [sys.executable, '-u', '-m', group['module'], *args]}


def preflight(catalogue, rows, root=ROOT):
    """Validate all selected frozen inputs, including paired file hashes. No fitting/test scoring."""
    import numpy as np
    from data.aligned_sequences import (AlignedFeatureSource, check_metadata_hash,
                                       load_aligned_normalization, load_split)
    from data.native_sequences import NativePairSource, load_native_normalization
    from data.feature_matrix import prepare_features
    groups = {r['group'] for r in rows}
    needed = {catalogue['metadata'], catalogue['split'], catalogue['manifest']}
    for r in rows:
        arguments = catalogue['groups'][r['group']]['args']+r['args']
        needed.update(a for a in arguments if a in catalogue['inputs'])
    for name in sorted(needed):
        path = root/name
        if not path.is_file():
            raise FileNotFoundError(f'Missing input: {name}')
        expected = catalogue['inputs'][name]
        if name == catalogue['metadata']:
            check_metadata_hash(path, expected)
        elif sha(path) != expected:
            raise ValueError(f'Frozen input changed: {name}; do not silently replace data')
    split, metadata, ids, labels, indices = load_split(root/catalogue['metadata'], root/catalogue['split'])
    if {k: len(v) for k,v in indices.items()} != catalogue['counts']:
        raise ValueError('Frozen split counts differ from catalogue')
    for row in rows:
        if row['group'] != 'hgb':
            continue
        args = catalogue['groups']['hgb']['args']+row['args']
        main = root/args[args.index('--features')+1]
        extras = [root/args[args.index(flag)+1] for flag in ('--rhythm-features','--structure-features') if flag in args]
        x,y,feature_ids,names,_ = prepare_features(main,metadata,list(ids),extras)
        if x.shape != (len(ids),row['n_features']) or not np.isfinite(x).all() or not np.array_equal(y,labels):
            raise ValueError(f'Invalid features: {row["id"]}')
        for path in (main,*extras):
            with np.load(path,allow_pickle=False) as data:
                check_metadata_hash(root/catalogue['metadata'],str(data['metadata_sha256'].item()))
    source_args = (ids,labels,root/'datasets/features/cqt',root/'datasets/features/rhythm_windows',
                   root/catalogue['metadata'],root/catalogue['split'])
    sources=[]
    if 'early' in groups:
        source=AlignedFeatureSource(*source_args,manifest_path=root/catalogue['manifest'])
        load_aligned_normalization(root/'datasets/features/cqt_rhythm_common_v1_normalization.npz',
            source,root/catalogue['split'],root/catalogue['metadata'],split['parts']['train'])
        sources.append(source)
    if 'late' in groups:
        source=NativePairSource(*source_args,manifest_path=root/catalogue['manifest'])
        load_native_normalization(root/'datasets/features/cqt_rhythm_native_normalization.npz',
            source,root/catalogue['split'],root/catalogue['metadata'],split['parts']['train'])
        sources.append(source)
    for source in sources:
        for i,key in enumerate(ids,1):
            source.read(key)
            if i%1000 == 0:
                print(f'Preflight {type(source).__name__}: {i}/{len(ids)}',flush=True)
    return {'counts':catalogue['counts'], 'input_sha256':{n:sha(root/n) for n in sorted(needed)},
            'source_signatures':[s.source_signature for s in sources],
            'paired_files_checked':len(ids) if sources else 0, 'test_scored':False}


def verify_result(run, catalogue, seed):
    """Completion requires actual files, matching split/seed/members and recomputed metrics."""
    import numpy as np
    from reporting.metrics import evaluate
    path=Path(run)/'artifacts'
    config=read_json(path/'config.json');metrics=read_json(path/'metrics.json')
    predictions=read_json(path/'predictions_val.json')
    model=path/('model.joblib' if 'chosen_max_leaf_nodes' in config else 'model.pt')
    if not model.is_file() or model.stat().st_size == 0:
        raise ValueError('Missing model artifact')
    if metrics.get('test_evaluated') is not False or 'test' in metrics:
        raise ValueError('This suite is validation-only')
    if config.get('split_sha256') != catalogue['inputs'][catalogue['split']] or config.get('sizes') != catalogue['counts']:
        raise ValueError('Result uses different split')
    actual_seed=config.get('training',{}).get('seed',config.get('model_seed'))
    if actual_seed != seed:
        raise ValueError('Result seed differs')
    split=read_json(ROOT/catalogue['split'])
    truth=read_json(ROOT/catalogue['metadata'])
    keys=[r['id'] for r in predictions]
    if len(keys)!=len(set(keys)) or set(keys)!=set(split['parts']['val']):
        raise ValueError('Prediction IDs do not match validation split')
    if any(int(truth[r['id']]['ps'])!=r['true_grade'] for r in predictions):
        raise ValueError('Prediction labels differ')
    computed=evaluate([r['true_grade'] for r in predictions],[r['score'] for r in predictions])
    for key in ('n','kendall_tau_c','spearman','mae','mse','acc_pm1'):
        a,b=computed[key],metrics['validation'][key]
        if not (a is None and b is None) and (a is None or b is None or not np.isclose(a,b,rtol=1e-6,atol=1e-7)):
            raise ValueError(f'Metric mismatch: {key}')
    return computed


def verify_outputs(item):
    run=item['run']
    if '--max-epochs' in item['command']:
        for required in ('history.json','history.jsonl','best_checkpoint.pt','run_parameters.json'):
            if not (run/'artifacts'/required).is_file():
                raise ValueError(f'Missing CRNN output: {required}')
    elif not (run/'artifacts/validation_search.json').is_file():
        raise ValueError('Missing HGB validation search')


def source_hashes():
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'src').rglob('*.py'))}


def launch(item, catalogue, seed, provenance, threads=4):
    run=item['run']
    run.mkdir(parents=True,exist_ok=False)  # Atomic reservation; never overwrite/reuse a failed run.
    state={'experiment_id':item['experiment_id'],'group':item['group'],'seed':seed,
           'status':'running','started_at_utc':now(),'command':item['command'],
           'catalogue_sha256':sha(CATALOGUE),'inputs':provenance,
           'source_sha256':source_hashes(),
           'threads':threads,'python':sys.version}
    write_json(run/'run.json',state,exclusive=True)
    child=None
    try:
        env=dict(os.environ,PYTHONPATH=str(ROOT/'src')+os.pathsep+os.environ.get('PYTHONPATH',''),
                 OMP_NUM_THREADS=str(threads),MKL_NUM_THREADS=str(threads),PYTHONDONTWRITEBYTECODE='1')
        with (run/'train.log').open('x',encoding='utf-8') as log:
            child=subprocess.Popen(item['command'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            print(f'Started {item["experiment_id"]}; log: {run/"train.log"}',flush=True)
            returncode=child.wait()
        if returncode:
            raise RuntimeError(f'Training exited {returncode}; see {run/"train.log"}')
        verify_outputs(item)
        if source_hashes()!=state['source_sha256']:
            raise ValueError('Source files changed during training')
        state['validation']=verify_result(run,catalogue,seed)
        state['status']='completed';state['returncode']=0
    except BaseException as exc:
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        state['status']='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed'
        state['error']=f'{type(exc).__name__}: {exc}'
        state['returncode']=child.returncode if child else None
        raise
    finally:
        state['finished_at_utc']=now();write_json(run/'run.json',state)


def summarize(catalogue, root=ROOT):
    """Regenerate only summaries/, never README or archives. Preserve every attempt/seed."""
    base=root/'outputs'/catalogue['suite'];rows=[]
    for experiment in catalogue['experiments']:
        folder=base/experiment['group']/experiment['folder']
        attempts=sorted(folder.glob('seed_*')) if folder.is_dir() else []
        if not attempts:
            rows.append({'experiment_id':experiment['id'],'group':experiment['group'],'status':'not_started'})
        for run in attempts:
            if not run.is_dir():continue
            state=read_json(run/'run.json') if (run/'run.json').is_file() else {'status':'unmanaged'}
            metrics_path=run/'artifacts/metrics.json'
            rows.append({'experiment_id':experiment['id'],'group':experiment['group'],
                         'run':run.relative_to(base).as_posix(),'status':state['status'],
                         'seed':state.get('seed'),
                         'validation':read_json(metrics_path).get('validation') if metrics_path.is_file() else None})
    payload={'suite':catalogue['suite'],'generated_at_utc':now(),'runs':rows,
             'note':'run.json is the status authority. running may be stale after a hard crash; not auto-resumed.'}
    target=base/'summaries';target.mkdir(parents=True,exist_ok=True)
    write_json(target/'results.json',payload)
    lines=['# 重跑结果与进度','','来源：每次运行的run.json和artifacts/metrics.json。不扫描历史归档，不按高分隐藏失败。',
           'running在进程被强杀后可能残留；确认日志/进程后使用新run名重试。','']
    for group in catalogue['groups']:
        lines.extend(['## '+group,''])
        for row in (r for r in rows if r['group']==group):
            text=f'- **{row["experiment_id"]}** — {row["status"]}'
            if row.get('run'):text+=f' · `{row["run"]}`'
            metrics=row.get('validation')
            if metrics:
                text+=' · '+', '.join(f'{k}={metrics[k]:.4f}' if metrics.get(k) is not None else f'{k}=null'
                                     for k in ('kendall_tau_c','mae','mse','acc_pm1'))
            lines.append(text)
        lines.append('')
    (target/'RESULTS.md').write_text('\n'.join(lines).rstrip()+'\n',encoding='utf-8')
    return payload


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--execute',action='store_true');mode.add_argument('--check',action='store_true')
    mode.add_argument('--summarize',action='store_true');mode.add_argument('--dry-run',action='store_true')
    parser.add_argument('--group',choices=['all','hgb','early','late'],default='all')
    parser.add_argument('--experiments',nargs='+')
    parser.add_argument('--seed',type=int,default=42);parser.add_argument('--run-name',default='run_01')
    parser.add_argument('--device',choices=['cuda','cpu','mps','auto'],default='cuda')
    parser.add_argument('--num-workers',type=int,default=4);parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--skip-completed',action='store_true')
    args=parser.parse_args();catalogue=read_json(CATALOGUE)
    if args.threads<1:parser.error('--threads must be positive')
    if args.skip_completed and not args.execute:parser.error('--skip-completed requires --execute')
    if args.summarize:
        summarize(catalogue);print(ROOT/'outputs'/catalogue['suite']/'summaries/RESULTS.md');return
    rows=select(catalogue,args.group,args.experiments)
    items=[plan(catalogue,row,args.seed,args.run_name,args.device,args.num_workers) for row in rows]
    for item in items:print(item['experiment_id']+'\n  PYTHONPATH=src '+shlex.join(item['command']),flush=True)
    if not (args.check or args.execute):return
    pending=[]
    for item in items:
        if args.execute and item['run'].exists():
            state_path=item['run']/'run.json'
            state=read_json(state_path) if state_path.is_file() else {}
            if args.skip_completed and state.get('status')=='completed':
                if (state.get('catalogue_sha256') != sha(CATALOGUE) or state.get('command') != item['command']
                        or state.get('source_sha256') != source_hashes() or state.get('threads')!=args.threads):
                    raise ValueError('Completed run configuration/source differs; choose another run-name')
                verify_outputs(item)
                verify_result(item['run'],catalogue,args.seed);continue
            raise FileExistsError(f'Run already exists: {item["run"]}; use a new --run-name')
        pending.append(item)
    if not pending:return
    selected_ids={x['experiment_id'] for x in pending}
    provenance=preflight(catalogue,[r for r in rows if r['id'] in selected_ids])
    print(json.dumps({'preflight':'passed',**provenance},ensure_ascii=False),flush=True)
    if args.check:return
    if any(i['group']!='hgb' for i in pending):
        from training.crnn import choose_device
        choose_device(args.device)  # Check before starting any expensive fit.
    try:
        for item in pending:
            launch(item,catalogue,args.seed,provenance,args.threads)
            summarize(catalogue)
    finally:summarize(catalogue)


if __name__=='__main__':
    main()
