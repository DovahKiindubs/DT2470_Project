"""固定分组划分：同作曲家或同 YouTube 录音的曲目必须留在同一集合。"""
import re
import unicodedata
from urllib.parse import urlparse, parse_qs

import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def recording_id(url):
    url = str(url or '').strip()
    if not url:
        return None
    parsed = urlparse(url)
    host = (parsed.hostname or '').lower()
    parts = parsed.path.strip('/').split('/')
    if host in ('youtu.be', 'www.youtu.be'):
        value = parts[0]
    elif host == 'youtube.com' or host.endswith('.youtube.com') or host == 'youtube-nocookie.com':
        value = parse_qs(parsed.query).get('v', [''])[0]
        if not value and len(parts) >= 2 and parts[0] in ('embed', 'shorts', 'v'):
            value = parts[1]
    else:
        return None
    return value if re.fullmatch(r'[A-Za-z0-9_-]{11}', value) else None


def make_groups(ids, metadata):
    # 并查集处理“同作曲家”和“共用录音”两个分组条件的传递关系。
    parent = list(range(len(ids)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, key in enumerate(ids):
        item = metadata[key]
        raw = item.get('composer')
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(f'Missing composer: {key}')
        composer = ' '.join(unicodedata.normalize('NFKC', raw).casefold().split())
        tokens = [('composer', composer)]
        video = recording_id(item.get('youtube_link'))
        if video:
            tokens.append(('recording', video))
        for token in tokens:
            if token in seen:
                parent[find(i)] = find(seen[token])
            else:
                seen[token] = i
    return np.asarray([find(i) for i in range(len(ids))])


def make_split(ids, groups, seed=42):
    if len(np.unique(groups)) < 5:
        raise ValueError('Need at least five independent groups for this baseline split')
    index = np.arange(len(ids))
    train_val, test = next(GroupShuffleSplit(n_splits=1, test_size=0.2,
                                             random_state=seed).split(index, groups=groups))
    tr, va = next(GroupShuffleSplit(n_splits=1, test_size=0.25,
                                    random_state=seed).split(train_val, groups=groups[train_val]))
    return {name: sorted(str(ids[i]) for i in values) for name, values in
            [('train', train_val[tr]), ('val', train_val[va]), ('test', test)]}


def validate_split(parts, ids, groups):
    if set(parts) != {'train', 'val', 'test'}:
        raise ValueError('Split must have exactly train, val and test')
    lookup = {str(key): i for i, key in enumerate(ids)}
    if len(lookup) != len(ids):
        raise ValueError('Duplicate recording keys')
    indices, seen_ids, seen_groups = {}, set(), set()
    for name in ('train', 'val', 'test'):
        keys = parts[name]
        if not keys or len(keys) != len(set(keys)) or not set(keys) <= set(lookup):
            raise ValueError(f'Empty, duplicate or unknown keys in {name}')
        ix = np.asarray([lookup[key] for key in keys])
        group_set = set(groups[ix])
        if seen_ids & set(keys) or seen_groups & group_set:
            raise ValueError(f'Recording/composer group leakage in {name}')
        seen_ids.update(keys)
        seen_groups.update(group_set)
        indices[name] = ix
    if seen_ids != set(lookup):
        raise ValueError('Split does not cover the complete feature set')
    return indices
