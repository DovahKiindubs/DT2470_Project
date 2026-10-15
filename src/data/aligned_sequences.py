"""Shared CQT/rhythm alignment for preparation and training (no audio dependencies)."""
import hashlib
import io
import json
import unicodedata
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from features.cqt import ArrayUnpickler, ROOT, sha256
from data.split_data import make_groups, validate_split

RHYTHM_NAMES = ('onset_density', 'onset_strength_mean', 'onset_strength_std',
                'onset_strength_max', 'ioi_mean', 'ioi_std', 'ioi_cv',
                'tempo_bpm', 'tempo_confidence')


def digest(value):
    return hashlib.sha256(value).hexdigest()


def nfc(value):
    return unicodedata.normalize('NFC', str(value))


def check_metadata_hash(path, expected):
    # Only LF/CRLF equivalence is accepted; semantic changes still fail.
    raw = Path(path).read_bytes()
    lf = raw.replace(b'\r\n', b'\n')
    if expected not in {digest(raw), digest(lf), digest(lf.replace(b'\n', b'\r\n'))}:
        raise ValueError('Split/cache metadata mismatch (not a line-ending difference)')


def load_split(metadata_path, split_path):
    metadata = json.loads(Path(metadata_path).read_text(encoding='utf-8'))
    doc = json.loads(Path(split_path).read_text(encoding='utf-8'))
    check_metadata_hash(metadata_path, doc['metadata_sha256'])
    if len({nfc(k) for k in metadata}) != len(metadata):
        raise ValueError('Metadata Unicode key collision')
    metadata = {nfc(k): v for k, v in metadata.items()}
    parts = {part: [nfc(k) for k in keys] for part, keys in doc['parts'].items()}
    ids = np.asarray(sorted(k for keys in parts.values() for k in keys))
    if not set(ids) <= set(metadata):
        raise ValueError('Split contains unknown metadata keys')
    labels = []
    for key in ids:
        if str(metadata[key].get('ps')) not in {str(i) for i in range(11)}:
            raise ValueError(f'Invalid grade: {key}')
        labels.append(int(metadata[key]['ps']))
    indices = validate_split(parts, ids, make_groups(ids, metadata))
    doc = dict(doc, parts=parts)
    return doc, metadata, ids, np.asarray(labels), indices


def scan_files(folder, suffix):
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f'Missing feature directory: {folder}')
    found = {}
    for path in folder.glob(f'*{suffix}'):
        if path.name == 'index.npz':
            continue
        key = nfc(path.stem)
        if key in found:
            raise ValueError(f'Unicode filename collision: {key}')
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'Unsafe feature path: {path}')
        found[key] = path
    return found


def alignment_config(fps=5.0, offset=0.0):
    if not np.isfinite(fps) or fps <= 0 or not np.isfinite(offset):
        raise ValueError('CQT fps must be positive and time offset finite')
    return {'version': 'center_linear_valid_brackets_v1', 'cqt_fps': float(fps),
            'cqt_time_offset_seconds': float(offset), 'rhythm_time': 'window center',
            'boundary': 'crop CQT to window-center coverage; no extrapolation or stretching',
            'invalid': 'exclude invalid values from train stats; standardized mean imputation (0)',
            'ioi_valid': 'ioi_mean > 0',
            'tempo_valid': 'onset_density > 0 and onset_strength_mean > 0 and ioi_mean > 0',
            'confidence_valid': 'onset_density > 0 and onset_strength_mean > 0',
            'timebase_limit': 'nominal CQT fps and common recording origin assumed; not waveform-verified'}


def align_arrays(cqt, rhythm, starts, win_sec, config, end_times=None):
    """Return raw 97xT values and per-value validity on the original CQT grid."""
    cqt, rhythm, starts = np.asarray(cqt), np.asarray(rhythm), np.asarray(starts, dtype=float)
    if cqt.ndim != 2 or cqt.shape[0] != 88 or cqt.shape[1] == 0 or not np.isfinite(cqt).all():
        raise ValueError('CQT must be finite (88,T>0)')
    if rhythm.ndim != 2 or rhythm.shape[0] != 9 or rhythm.shape[1] == 0:
        raise ValueError('Rhythm must be (9,N>0)')
    if starts.shape != (rhythm.shape[1],) or not np.isfinite(starts).all() or not np.isfinite(rhythm).all():
        raise ValueError('Invalid rhythm values/timestamps')
    if starts[0] < 0 or np.any(np.diff(starts) <= 0) or np.any(rhythm < 0):
        raise ValueError('Rhythm must be nonnegative and timestamps strictly increasing')
    if not np.isfinite(win_sec) or win_sec <= 0:
        raise ValueError('Invalid rhythm window length')
    if end_times is None:
        if len(starts) == 1:
            raise ValueError('Single rhythm window needs window_end_times; actual short-window length is unknown')
        centers = starts + win_sec / 2.0
    else:
        ends = np.asarray(end_times, dtype=float)
        if ends.shape != starts.shape or not np.isfinite(ends).all() or np.any(ends <= starts):
            raise ValueError('Invalid window_end_times')
        centers = (starts + ends) / 2.0
    if np.any(np.diff(centers) <= 0):
        raise ValueError('Window centers must increase')
    fps, offset = config['cqt_fps'], config['cqt_time_offset_seconds']
    times = np.arange(cqt.shape[1], dtype=float) / fps + offset
    selected = np.flatnonzero((times >= centers[0] - 1e-9) & (times <= centers[-1] + 1e-9))
    if selected.size == 0:
        raise ValueError('No CQT frames in rhythm-center coverage')
    target = np.clip(times[selected], centers[0], centers[-1])
    valid = np.ones(rhythm.shape, dtype=bool)
    valid[4:7] = rhythm[4] > 0
    active = (rhythm[0] > 0) & (rhythm[1] > 0)
    valid[7] = active & (rhythm[4] > 0)
    valid[8] = active
    # Invalid windows must not be bridged/interpolated as real observations.
    right = np.searchsorted(centers, target, side='left').clip(0, len(centers)-1)
    left = np.maximum(right-1, 0)
    exact_right = np.isclose(target, centers[right], rtol=0, atol=1e-9)
    exact_left = np.isclose(target, centers[left], rtol=0, atol=1e-9)
    right[exact_left] = left[exact_left]
    left[exact_right] = right[exact_right]
    exact = exact_left | exact_right
    target[exact] = centers[right[exact]]
    aligned_valid = valid[:, left] & valid[:, right]
    aligned = np.vstack([np.interp(target, centers, row) for row in rhythm])
    aligned[~aligned_valid] = 0.0
    values = np.concatenate((cqt[:, selected], aligned), axis=0)
    masks = np.concatenate((np.ones((88, selected.size), dtype=bool), aligned_valid), axis=0)
    return values, masks


class AlignedFeatureSource:
    """Resolve filenames once and verify immutable pair hashes from the freeze manifest."""
    def __init__(self, ids, labels, cqt_dir, rhythm_dir, metadata_path,
                 split_path, fps=5.0, offset=0.0, manifest_path=None):
        self.config = alignment_config(fps, offset)
        self.cqt = scan_files(cqt_dir, '.bin')
        self.rhythm = scan_files(rhythm_dir, '.npz')
        self.labels = {nfc(k): int(y) for k, y in zip(ids, labels)}
        if len(self.labels) != len(ids):
            raise ValueError('Duplicate sequence keys')
        missing = set(self.labels) - (set(self.cqt) & set(self.rhythm))
        if missing:
            raise ValueError(f'Missing paired files ({len(missing)}), e.g. {sorted(missing)[:3]}')
        index_path = Path(rhythm_dir) / 'index.npz'
        with np.load(index_path, allow_pickle=False) as idx:
            index_ids = [nfc(k) for k in idx['ids']]
            if len(index_ids) != len(set(index_ids)):
                raise ValueError('Duplicate rhythm index keys')
            if idx['feature_names'].tolist() != list(RHYTHM_NAMES):
                raise ValueError('Unexpected rhythm channel order')
            check_metadata_hash(metadata_path, str(idx['metadata_sha256'].item()))
            self.win = float(idx['win_sec']); self.hop = float(idx['hop_sec'])
            if not np.isfinite([self.win, self.hop]).all() or min(self.win, self.hop) <= 0:
                raise ValueError('Invalid index window parameters')
            self.index = {key: (int(idx['y'][i]), int(idx['lengths'][i]), str(idx['audio_sha256'][i]))
                          for i, key in enumerate(index_ids)}
        if not set(self.labels) <= set(self.index):
            raise ValueError('Frozen keys missing from rhythm index')
        doc = json.loads(Path(split_path).read_text(encoding='utf-8'))
        if manifest_path is None and doc.get('manifest'):
            manifest_path = ROOT / doc['manifest']
        self.expected = {}
        self.manifest_sha256 = ''
        if manifest_path is not None:
            manifest_path = Path(manifest_path)
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            if manifest['split_sha256'] != sha256(split_path):
                raise ValueError('Manifest does not match frozen split')
            if manifest['source_sha256']['datasets/features/rhythm_windows/index.npz'] != sha256(index_path):
                raise ValueError('Rhythm index changed since freeze')
            records = {nfc(r['id']): r for r in manifest['records']}
            for key, label in self.labels.items():
                r = records.get(key)
                if r is None or r['status'] != 'included' or int(r['label']) != label:
                    raise ValueError(f'Not an included manifest member: {key}')
                self.expected[key] = (r['cqt_sha256'], r['rhythm_sha256'])
            self.manifest_sha256 = sha256(manifest_path)
        else:
            # Useful for temporary test datasets or explicitly supplied non-frozen splits.
            self.expected = {k: (sha256(self.cqt[k]), sha256(self.rhythm[k])) for k in self.labels}
        identity = {'pairs': sorted((k, *v) for k, v in self.expected.items()),
                    'index_sha256': sha256(index_path)}
        self.source_signature = digest(json.dumps(identity, ensure_ascii=False, sort_keys=True).encode())

    def read(self, key):
        key = nfc(key)
        cb, rb = self.cqt[key].read_bytes(), self.rhythm[key].read_bytes()
        if (digest(cb), digest(rb)) != self.expected[key]:
            raise ValueError(f'Feature file changed since freeze/preparation: {key}')
        cqt = ArrayUnpickler(io.BytesIO(cb)).load()
        with np.load(io.BytesIO(rb), allow_pickle=False) as r:
            x, times = r['X'], r['times']
            label, length, audio_hash = self.index[key]
            if (label != self.labels[key] or int(r['y']) != label or x.shape != (9, length)
                    or str(r['audio_sha256'].item()) != audio_hash
                    or r['feature_names'].tolist() != list(RHYTHM_NAMES)
                    or float(r['win_sec']) != self.win or float(r['hop_sec']) != self.hop):
                raise ValueError(f'Rhythm file/index mismatch: {key}')
            ends = r['window_end_times'] if 'window_end_times' in r.files else None
            return align_arrays(cqt, x, times, self.win, self.config, ends)


def compute_aligned_statistics(source, train_ids):
    # Channel-wise parallel variance merge (stable even for large offsets).
    counts = np.zeros(97, dtype=np.int64)
    mean = np.zeros(97, dtype=np.float64)
    m2 = np.zeros(97, dtype=np.float64)
    frames = 0
    for i, key in enumerate(train_ids, 1):
        x, valid = source.read(key)
        frames += x.shape[1]
        for channel in range(97):
            values = x[channel, valid[channel]].astype(np.float64)
            n = values.size
            if n == 0:
                continue
            local_mean = values.mean()
            delta = local_mean - mean[channel]
            total = counts[channel] + n
            m2[channel] += np.square(values-local_mean).sum() + delta**2 * counts[channel]*n/total
            mean[channel] += delta*n/total
            counts[channel] = total
        if i % 500 == 0:
            print(f'Aligned train statistics {i}/{len(train_ids)}', flush=True)
    if frames == 0:
        raise ValueError('Empty training data')
    variance = np.divide(m2, counts, out=np.zeros(97), where=counts > 0)
    std = np.sqrt(variance)
    # Constant/all-invalid channels remain finite and do not amplify tiny noise.
    std[std < 1e-8] = 1.0
    return mean, std, counts, frames


class AlignedSequenceDataset(Dataset):
    def __init__(self, ids, labels, source, mean, std, include_rhythm):
        self.ids = list(ids)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.source = source
        self.channels = 97 if include_rhythm else 88
        self.mean = np.asarray(mean, dtype=np.float32).reshape(97, 1)
        self.std = np.asarray(std, dtype=np.float32).reshape(97, 1)
        if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or np.any(self.std <= 0):
            raise ValueError('Invalid aligned normalization')
        self.lengths = [source.read(k)[0].shape[1] for k in ids]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        x, valid = self.source.read(self.ids[index])
        x = ((x.astype(np.float32)-self.mean)/self.std)
        x[~valid] = 0.0  # Mean imputation in standardized space; no extra feature channels.
        if not np.isfinite(x).all():
            raise ValueError(f'Non-finite standardized sequence: {self.ids[index]}')
        return torch.from_numpy(x[:self.channels].copy()), torch.tensor(self.labels[index]), self.ids[index]


def load_aligned_normalization(path, source, split_path, metadata_path, train_ids):
    with np.load(path, allow_pickle=False) as d:
        required = {'mean', 'std', 'valid_counts', 'preprocessing_json', 'source_signature',
                    'split_sha256', 'metadata_sha256', 'train_ids_sha256'}
        if not required <= set(d.files):
            raise ValueError('Need newly prepared aligned normalization, not legacy CQT-only cache')
        expected_train = digest('\n'.join(train_ids).encode())
        if (str(d['split_sha256'].item()) != sha256(split_path)
                or str(d['train_ids_sha256'].item()) != expected_train
                or str(d['source_signature'].item()) != source.source_signature
                or json.loads(str(d['preprocessing_json'].item())) != source.config):
            raise ValueError('Normalization split, sources, train ids or alignment configuration mismatch')
        check_metadata_hash(metadata_path, str(d['metadata_sha256'].item()))
        mean, std = d['mean'], d['std']
        if mean.shape != (97,) or std.shape != (97,) or d['valid_counts'].shape != (97,):
            raise ValueError('Aligned normalization must have 97 channels')
        return mean.copy(), std.copy()
