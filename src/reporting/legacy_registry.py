"""Build non-destructive reports for every experiment under outputs/."""
import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from features.cqt import ROOT,sha256

METRICS=('kendall_tau_c','spearman','mae','mse','acc_pm1')


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def infer_family(rel,config):
    if config:
        return str(config.get('model') or config.get('feature_group') or 'unknown')
    return 'unknown'


def scan(outputs):
    rows=[]
    for metric_path in sorted(outputs.rglob('metrics.json')):
        run=metric_path.parent;rel=run.relative_to(outputs).as_posix()
        metrics=load(metric_path);val=metrics.get('validation',{})
        config_path=run/'config.json';config=load(config_path) if config_path.exists() else {}
        split_path=config.get('split_file')
        split_sha=config.get('split_sha256')
        training=config.get('training',{}) if isinstance(config.get('training'),dict) else {}
        files=sorted(p.name for p in run.iterdir() if p.is_file())
        complete=all((run/name).exists() for name in ('metrics.json','config.json','predictions_val.json'))
        row={'run':rel,'status':'complete' if complete else 'partial','model':infer_family(rel,config),
             'feature_group':config.get('feature_group'), 'n_validation':val.get('n'),
             **{k:val.get(k) for k in METRICS},'test_evaluated':bool(metrics.get('test_evaluated')),
             'split_sha256':split_sha,'split_file':split_path,'seed':training.get('seed'),
             'best_epoch':training.get('best_epoch'),'epochs_run':training.get('epochs_run'),
             'runtime_seconds':config.get('runtime_seconds'),'config_sha256':sha256(config_path) if config_path.exists() else None,
             'metrics_sha256':sha256(metric_path),'files':files}
        rows.append(row)
    return rows


def comparable(rows):
    groups={}
    for row in rows:
        key=(row['split_sha256'] or 'unknown',row['n_validation'])
        groups.setdefault(key,[]).append(row['run'])
    return [{'split_sha256':k[0],'n_validation':k[1],'n_runs':len(v),'runs':v}
            for k,v in sorted(groups.items(),key=lambda x:(str(x[0][0]),str(x[0][1])))]


def table(rows):
    def cell(value,digits=4):
        if value is None:return '—'
        if isinstance(value,float):return f'{value:.{digits}f}'
        return str(value)
    out=['| Run | N | Tau-c | Spearman | MAE | MSE | Acc±1 | Test |',
         '|---|---:|---:|---:|---:|---:|---:|---|']
    for r in sorted(rows,key=lambda x:(-(x['kendall_tau_c'] if x['kendall_tau_c'] is not None else -999),x['run'])):
        out.append('| '+ ' | '.join([f'`{r["run"]}`',cell(r['n_validation']),cell(r['kendall_tau_c']),
            cell(r['spearman']),cell(r['mae']),cell(r['mse']),cell(r['acc_pm1']),
            'yes' if r['test_evaluated'] else 'no'])+' |')
    return '\n'.join(out)


def build(outputs):
    rows=scan(outputs);groups=comparable(rows);summary=outputs/'summaries';summary.mkdir(parents=True,exist_ok=True)
    generated=datetime.now(timezone.utc).isoformat()
    payload={'generated_at_utc':generated,'root':str(outputs.resolve()),'n_runs':len(rows),
             'comparability_rule':'Compare performance only when split_sha256 and validation N are identical. Missing split SHA means historical/unknown provenance.',
             'runs':rows,'comparison_groups':groups}
    (summary/'experiment_index.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    columns=['run','status','model','feature_group','n_validation',*METRICS,'test_evaluated','split_sha256',
             'seed','best_epoch','epochs_run','runtime_seconds','config_sha256','metrics_sha256']
    with (summary/'experiment_index.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for row in rows:writer.writerow({k:row.get(k) for k in columns})
    full=[r for r in rows if r['n_validation']==2095]
    audio=[r for r in rows if r['n_validation']==2044]
    common=[r for r in rows if r['n_validation']==2040]
    readme=f'''# Experiment registry

Generated from existing run artifacts; no experiment directory is moved, renamed, deleted, or overwritten.

- Total discovered runs: **{len(rows)}**
- Test evaluated runs: **{sum(r['test_evaluated'] for r in rows)}**
- Full strict split validation runs (N=2095): **{len(full)}**
- Audio subset validation runs (N=2044): **{len(audio)}**
- Frozen CQT/rhythm common-v1 validation runs (N=2040): **{len(common)}**

## Comparison rule

Only compare model metrics directly when both `split_sha256` and validation N match. Runs from N=2095, 2044, and 2040 answer different experiments. All current rows are validation results; test remains sealed.

## Frozen common-v1 CRNN family (N=2040)

{table(common) if common else 'No completed runs yet.'}

## Full strict split family (N=2095)

{table(full)}

## Audio subset feature ablations (N=2044)

{table(audio)}

## Files

- `experiment_index.json`: complete machine-readable registry, file lists, hashes, and comparison groups.
- `experiment_index.csv`: report-friendly flat table.
- `EXPERIMENT_LOG.md`: human-curated chronology, conclusions, caveats, and next planned runs.
- Run directories remain the source of truth for models, predictions, configs, and histories.

Refresh after copying new results:

```bash
python src/summarize_experiments.py
```

Generated UTC: {generated}
'''
    (outputs/'README.md').write_text(readme,encoding='utf-8')
    return payload


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--outputs',type=Path,default=ROOT/'outputs')
    args=p.parse_args();result=build(args.outputs);print(json.dumps({'n_runs':result['n_runs'],'output':str(args.outputs)},indent=2))
