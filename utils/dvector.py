import numpy as np
import torch


def get_model_value(hp, name, default):
    return hp.model.get(name, default)


def use_d_vector(hp):
    value = get_model_value(hp, 'use_d_vector', True)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ('1', 'true', 'yes', 'on'):
            return True
        if normalized in ('0', 'false', 'no', 'off'):
            return False
    return bool(value)


def zero_embedding_numpy(hp):
    return np.zeros((hp.embedder.emb_dim,), dtype=np.float32)


def zero_embedding(batch_size, hp, device=None, dtype=torch.float32):
    return torch.zeros((batch_size, hp.embedder.emb_dim), device=device, dtype=dtype)
