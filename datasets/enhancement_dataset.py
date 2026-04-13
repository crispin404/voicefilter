import random

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.audio import Audio, load_wav, pad_or_trim_wav
from utils.dataset_index import load_jsonl


class EnhancementDataset(Dataset):
    def __init__(self, manifest_path, hp, train=True):
        self.items = load_jsonl(manifest_path)
        self.hp = hp
        self.train = train
        self.audio = Audio(hp)
        self.segment_length = int(round(hp.data.segment_seconds * hp.audio.sample_rate))

    def __len__(self):
        return len(self.items)

    def _crop_aligned_pair(self, clean_wav, mix_wav):
        pair_length = min(len(clean_wav), len(mix_wav))
        clean_wav = clean_wav[:pair_length]
        mix_wav = mix_wav[:pair_length]

        if self.segment_length <= 0:
            return clean_wav, mix_wav

        if pair_length <= self.segment_length:
            return (
                pad_or_trim_wav(clean_wav, self.segment_length),
                pad_or_trim_wav(mix_wav, self.segment_length),
            )

        if self.train:
            start = random.randint(0, pair_length - self.segment_length)
        else:
            start = max(0, (pair_length - self.segment_length) // 2)
        end = start + self.segment_length
        return clean_wav[start:end], mix_wav[start:end]

    def __getitem__(self, idx):
        item = self.items[idx]
        clean_wav, _ = load_wav(item['clean_path'], sample_rate=self.hp.audio.sample_rate, mono=True)
        mix_wav, _ = load_wav(item['mix_path'], sample_rate=self.hp.audio.sample_rate, mono=True)
        clean_wav, mix_wav = self._crop_aligned_pair(clean_wav, mix_wav)

        clean_mag, _ = self.audio.wav2spec(clean_wav)
        mix_mag, _ = self.audio.wav2spec(mix_wav)
        embedding = np.load(item['embedding_path']).astype(np.float32)

        return {
            'subject_id': item['subject_id'],
            'clean_mag': torch.from_numpy(clean_mag).float(),
            'mix_mag': torch.from_numpy(mix_mag).float(),
            'embedding': torch.from_numpy(embedding).float(),
        }


def enhancement_collate_fn(batch):
    return {
        'subject_id': [item['subject_id'] for item in batch],
        'clean_mag': torch.stack([item['clean_mag'] for item in batch], dim=0),
        'mix_mag': torch.stack([item['mix_mag'] for item in batch], dim=0),
        'embedding': torch.stack([item['embedding'] for item in batch], dim=0),
    }
