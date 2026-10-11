"""对齐 MP3 与元数据、批量质检；写出 audio_qc.json 与 audio_composer_subset.json。"""
import json
from pathlib import Path

from audio_io import (
    PEAK_TARGET,
    AudioLoadError,
    duration_rel_err,
    load_audio,
    peak_normalize,
    qc_flags,
    waveform_stats,
)

ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIR = ROOT / 'datasets' / 'audio' / 'mp3_pieces'
METADATA = ROOT / 'datasets' / 'metadata' / 'new_clean_data.json'
SPLIT = ROOT / 'datasets' / 'splits' / 'cqt_composer_split.json'
REPORT = ROOT / 'datasets' / 'audio' / 'reports' / 'audio_qc.json'
SUBSET = ROOT / 'datasets' / 'splits' / 'audio_composer_subset.json'


def _rel(path):
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT)).replace('\\', '/')
    except ValueError:
        return str(path).replace('\\', '/')


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def build_inventory(audio_dir=AUDIO_DIR, metadata_path=METADATA):
    audio_dir, metadata_path = Path(audio_dir), Path(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    if not isinstance(metadata, dict) or not metadata:
        raise ValueError('Metadata must be a nonempty object')

    files = {p.stem: p for p in sorted(audio_dir.glob('*.mp3')) if not p.is_symlink()}
    meta_keys, audio_keys = set(metadata), set(files)
    matched = sorted(meta_keys & audio_keys)
    missing = sorted(meta_keys - audio_keys)
    orphan = sorted(audio_keys - meta_keys)

    matched_items = [{
        'key': key,
        'path': _rel(files[key]),
        'bytes': int(files[key].stat().st_size),
        'meta_duration': metadata[key].get('duration'),
        'ps': str(metadata[key].get('ps')),
    } for key in matched]

    inventory = {
        'audio_dir': _rel(audio_dir),
        'metadata_path': _rel(metadata_path),
        'n_metadata': len(meta_keys),
        'n_audio_files': len(audio_keys),
        'exact_match': len(matched),
        'missing_audio': len(missing),
        'orphan_audio': len(orphan),
        'missing_audio_keys': missing,
        'orphan_audio_keys': orphan,
        'matched': matched_items,
    }
    return inventory, metadata, files


def _meta_duration(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def qc_one(key, path, meta_duration):
    abs_path = Path(path)
    if not abs_path.is_absolute():
        abs_path = ROOT / abs_path
    try:
        rel_path = str(abs_path.resolve().relative_to(ROOT)).replace('\\', '/')
    except ValueError:
        rel_path = str(abs_path).replace('\\', '/')
    record = {
        'key': key, 'path': rel_path,
        'bytes': int(abs_path.stat().st_size) if abs_path.is_file() else None,
        'sr_out': None, 'n_samples': None, 'duration_sec': None,
        'peak': None, 'rms': None, 'clip_fraction': None,
        'meta_duration': meta_duration, 'duration_rel_err': None,
        'flags': [], 'ok': False,
    }
    try:
        y, sr_out = load_audio(abs_path, normalize=False)
        stats = waveform_stats(y, sr=sr_out)
        hard, soft = qc_flags(stats, meta_duration=meta_duration)
        if 'silence' not in hard and 'non_finite' not in hard:
            peak_normalize(y)
        record.update({
            'sr_out': int(sr_out),
            'n_samples': stats['n_samples'],
            'duration_sec': stats['duration_sec'],
            'peak': stats['peak'],
            'rms': stats['rms'],
            'clip_fraction': stats['clip_fraction'],
            'duration_rel_err': duration_rel_err(stats['duration_sec'], meta_duration),
            'flags': hard + soft,
            'ok': len(hard) == 0,
        })
    except AudioLoadError as exc:
        record['flags'] = ['decode_error']
        record['error'] = str(exc)
    except Exception as exc:
        record['flags'] = ['decode_error']
        record['error'] = f'{type(exc).__name__}: {exc}'
    return record


def build_common_subset(valid_keys, split_path=SPLIT):
    split = json.loads(Path(split_path).read_text(encoding='utf-8'))
    parts = split['parts']
    if set(parts) != {'train', 'val', 'test'}:
        raise ValueError('Split must contain train, val, test')
    valid = set(valid_keys)
    subset_parts, counts, source_counts, missing_parts, missing_counts = {}, {}, {}, {}, {}
    for name in ('train', 'val', 'test'):
        source_keys = list(parts[name])
        source_counts[name] = len(source_keys)
        subset_parts[name] = sorted(k for k in source_keys if k in valid)
        missing_parts[name] = sorted(k for k in source_keys if k not in valid)
        counts[name] = len(subset_parts[name])
        missing_counts[name] = len(missing_parts[name])

    train, val, test = map(set, (subset_parts['train'], subset_parts['val'], subset_parts['test']))
    if train & val or train & test or val & test:
        raise ValueError('Common subset parts overlap')

    return {
        'source_split': _rel(split_path),
        'split_seed': split.get('seed'),
        'n_valid': len(valid),
        'source_counts': source_counts,
        'counts': counts,
        'n_total': sum(counts.values()),
        'missing_counts': missing_counts,
        'n_missing': sum(missing_counts.values()),
        'note': ('Intersection of QC-valid audio with frozen composer-disjoint split. '
                 'missing_vs_source: in source_split but without QC-valid audio.'),
        'parts': subset_parts,
        'missing_vs_source': missing_parts,
    }


def _compact_ok(row):
    return {'key': row['key'], 'flags': list(row.get('flags') or [])}


def assemble_report(inventory, rows_by_key, matched_keys):
    records = [rows_by_key[k] for k in matched_keys if k in rows_by_key]
    failed = [r for r in records if not r.get('ok')]
    ok = sorted((_compact_ok(r) for r in records if r.get('ok')), key=lambda x: x['key'])
    flag_counts = {}
    for r in records:
        for flag in r.get('flags') or []:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
    return {
        'sr_policy': 'native',
        'peak_target': PEAK_TARGET,
        'inventory': {
            'audio_dir': inventory['audio_dir'],
            'metadata_path': inventory['metadata_path'],
            'n_metadata': inventory['n_metadata'],
            'n_audio_files': inventory['n_audio_files'],
            'exact_match': inventory['exact_match'],
            'missing_audio': inventory['missing_audio'],
            'orphan_audio': inventory['orphan_audio'],
            'missing_audio_keys': inventory['missing_audio_keys'],
            'orphan_audio_keys': inventory['orphan_audio_keys'],
        },
        'n_queued': len(records),
        'n_ok': len(ok),
        'n_failed': len(failed),
        'flag_counts': dict(sorted(flag_counts.items())),
        'failed': failed,
        'ok': ok,
    }


def _load_existing_status(report_path):
    existing = {}
    if not Path(report_path).is_file():
        return existing
    prev = json.loads(Path(report_path).read_text(encoding='utf-8'))
    for row in prev.get('ok') or []:
        existing[row['key']] = {'key': row['key'], 'ok': True, 'flags': list(row.get('flags') or [])}
    for row in prev.get('failed') or []:
        existing[row['key']] = row
    return existing


def run_qc(inventory, report_path=REPORT, *, resume=True, force_keys=None, limit=None):
    """limit 仅供测试截断；正式运行不要传。"""
    report_path = Path(report_path)
    existing = _load_existing_status(report_path) if resume else {}
    for key in set(force_keys or []):
        existing.pop(key, None)

    matched = inventory['matched'][:limit] if limit is not None else inventory['matched']
    matched_keys = [item['key'] for item in matched]
    rows_by_key, pending = {}, []
    for item in matched:
        key = item['key']
        cached = existing.get(key)
        if cached is not None and cached.get('flags') is not None:
            rows_by_key[key] = cached
        else:
            pending.append(item)

    def _flush():
        payload = assemble_report(inventory, rows_by_key, matched_keys)
        _write_json(report_path, payload)
        return payload

    for i, item in enumerate(pending, 1):
        row = qc_one(item['key'], ROOT / item['path'], _meta_duration(item.get('meta_duration')))
        row['path'] = item['path']
        row['bytes'] = item['bytes']
        rows_by_key[row['key']] = row
        if i % 25 == 0 or i == len(pending):
            _flush()
            print(f'QC {i}/{len(pending)}', flush=True)
    return _flush()


def main():
    inventory, _, _ = build_inventory()
    print(f'Inventory: match={inventory["exact_match"]} '
          f'missing={inventory["missing_audio"]} orphan={inventory["orphan_audio"]}', flush=True)
    payload = run_qc(inventory)
    subset = build_common_subset([r['key'] for r in payload['ok']])
    _write_json(SUBSET, subset)
    print(f'QC ok={payload["n_ok"]} failed={payload["n_failed"]} -> {REPORT}', flush=True)
    print(f'Subset {subset["counts"]} missing={subset["n_missing"]} -> {SUBSET}', flush=True)


if __name__ == '__main__':
    main()
