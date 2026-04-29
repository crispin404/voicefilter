import random

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.audio import Audio, compute_rms, load_wav, pad_or_trim_wav
from utils.dataset_index import load_jsonl
from utils.dvector import use_d_vector, zero_embedding_numpy


def get_data_value(hp, name, default):
    return hp.data.get(name, default)


class EnhancementDataset(Dataset):
    def __init__(self, manifest_path, hp, train=True):
        self.items = load_jsonl(manifest_path)
        self.hp = hp
        self.train = train
        self.audio = Audio(hp)
        self.segment_length = int(round(hp.data.segment_seconds * hp.audio.sample_rate))
        self.active_crop_enabled = bool(get_data_value(hp, 'active_crop_enabled', False))
        self.active_crop_probability = float(get_data_value(hp, 'active_crop_probability', 0.8))
        self.active_crop_trials = int(get_data_value(hp, 'active_crop_trials', 12))
        self.active_crop_min_rms = float(get_data_value(hp, 'active_crop_min_rms', 1e-4))
        self.active_crop_relative_rms = float(get_data_value(hp, 'active_crop_relative_rms', 0.5))
        self.use_d_vector = use_d_vector(hp)

    def __len__(self):
        return len(self.items)

    def _random_start(self, pair_length):
        return random.randint(0, pair_length - self.segment_length)

    def _active_start(self, clean_wav, pair_length):
        if (
            not self.active_crop_enabled
            or random.random() >= self.active_crop_probability
            or self.active_crop_trials <= 0
        ):
            return self._random_start(pair_length)

        full_clean_rms = compute_rms(clean_wav)
        active_threshold = max(self.active_crop_min_rms, full_clean_rms * self.active_crop_relative_rms)
        best_start = 0
        best_rms = -1.0

        for _ in range(self.active_crop_trials):
            start = self._random_start(pair_length)
            end = start + self.segment_length
            window_rms = compute_rms(clean_wav[start:end])
            if window_rms >= active_threshold:
                return start
            if window_rms > best_rms:
                best_rms = window_rms
                best_start = start

        return best_start

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
            start = self._active_start(clean_wav, pair_length)
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
        if self.use_d_vector:
            embedding = np.load(item['embedding_path']).astype(np.float32)
        else:
            embedding = zero_embedding_numpy(self.hp)

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
