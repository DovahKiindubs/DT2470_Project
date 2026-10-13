"""轻量 CQT CRNN：卷积、双向 GRU、掩码注意力与连续难度回归。"""
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, Sampler

from features_cqt import ArrayUnpickler


def downsample_lengths(lengths):
    lengths = torch.as_tensor(lengths, dtype=torch.long)
    # 两个 Conv1d(kernel=5, stride=2, padding=2)：ceil(length / 2) 两次。
    return torch.div(torch.div(lengths+1, 2, rounding_mode='floor')+1, 2, rounding_mode='floor')


class CQTSequenceDataset(Dataset):
    def __init__(self, ids, labels, cqt_dir, mean, std):
        self.ids = list(ids)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.cqt_dir = Path(cqt_dir)
        self.mean = np.asarray(mean, dtype=np.float32).reshape(88, 1)
        self.std = np.asarray(std, dtype=np.float32).reshape(88, 1)
        self.lengths = []
        for key in self.ids:
            path = self.cqt_dir/f'{key}.bin'
            if not path.is_file() or path.is_symlink():
                raise ValueError(f'Missing or unsafe CQT: {path}')
            with path.open('rb') as f:
                cqt = ArrayUnpickler(f).load()
            if cqt.ndim != 2 or cqt.shape[0] != 88 or cqt.shape[1] == 0:
                raise ValueError(f'Bad CQT shape for {key}: {cqt.shape}')
            self.lengths.append(cqt.shape[1])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        path = self.cqt_dir/f'{self.ids[index]}.bin'
        with path.open('rb') as f:
            cqt = ArrayUnpickler(f).load().astype(np.float32, copy=False)
        if not np.isfinite(cqt).all():
            raise ValueError(f'Non-finite CQT: {path}')
        cqt = (cqt-self.mean)/self.std
        return torch.from_numpy(cqt), torch.tensor(self.labels[index]), self.ids[index]


def collate_sequences(batch):
    sequences, labels, ids = zip(*batch)
    lengths = torch.tensor([x.shape[1] for x in sequences], dtype=torch.long)
    padded = torch.zeros(len(sequences), 88, int(lengths.max()), dtype=torch.float32)
    for i, sequence in enumerate(sequences):
        padded[i, :, :sequence.shape[1]] = sequence
    return padded, lengths, torch.stack(labels), list(ids)


class LengthBucketBatchSampler(Sampler):
    """每个 epoch 仅打乱相邻长度桶，降低变长序列 padding 浪费。"""
    def __init__(self, lengths, batch_size, seed=42, shuffle=True, bucket_multiplier=20):
        self.lengths = np.asarray(lengths)
        self.batch_size, self.seed, self.shuffle = batch_size, seed, shuffle
        self.bucket_size = batch_size*bucket_multiplier
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return (len(self.lengths)+self.batch_size-1)//self.batch_size

    def __iter__(self):
        ordered = np.argsort(self.lengths, kind='stable')
        chunks = [ordered[i:i+self.bucket_size].copy() for i in range(0,len(ordered),self.bucket_size)]
        rng = np.random.default_rng(self.seed+self.epoch)
        if self.shuffle:
            for chunk in chunks: rng.shuffle(chunk)
            rng.shuffle(chunks)
        indices = np.concatenate(chunks)
        for i in range(0,len(indices),self.batch_size):
            yield indices[i:i+self.batch_size].tolist()


class CQTCRNN(nn.Module):
    def __init__(self, conv_channels=64, hidden_size=64, dropout=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(88, conv_channels, 5, stride=2, padding=2),
            nn.BatchNorm1d(conv_channels), nn.GELU(),
            nn.Conv1d(conv_channels, conv_channels, 5, stride=2, padding=2),
            nn.BatchNorm1d(conv_channels), nn.GELU(), nn.Dropout(dropout),
        )
        self.gru = nn.GRU(conv_channels, hidden_size, batch_first=True,
                          bidirectional=True)
        self.attention = nn.Linear(hidden_size*2, 1)
        self.head = nn.Sequential(nn.LayerNorm(hidden_size*2), nn.Dropout(dropout),
                                  nn.Linear(hidden_size*2, 1))

    def forward(self, x, lengths, return_attention=False):
        encoded = self.conv(x).transpose(1, 2)
        reduced = downsample_lengths(lengths).clamp(max=encoded.shape[1])
        packed = pack_padded_sequence(encoded, reduced.cpu(), batch_first=True,
                                      enforce_sorted=False)
        packed_output, _ = self.gru(packed)
        output, _ = pad_packed_sequence(packed_output, batch_first=True,
                                        total_length=encoded.shape[1])
        positions = torch.arange(output.shape[1], device=output.device)[None, :]
        mask = positions < reduced.to(output.device)[:, None]
        logits = self.attention(output).squeeze(-1).masked_fill(~mask, -torch.inf)
        weights = torch.softmax(logits, dim=1)
        pooled = torch.sum(output*weights.unsqueeze(-1), dim=1)
        score = self.head(pooled).squeeze(-1)
        return (score, weights, reduced) if return_attention else score
