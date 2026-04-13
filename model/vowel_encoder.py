import torch
import torch.nn as nn

from .embedder import SpeechEmbedder


class VowelEmbeddingEncoder(nn.Module):
    def __init__(self, hp):
        super(VowelEmbeddingEncoder, self).__init__()
        self.embedder = SpeechEmbedder(hp)
        self.hp = hp

    def load_embedder(self, checkpoint_path=None, strict=True):
        if checkpoint_path is None:
            return False
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        self.embedder.load_state_dict(state_dict, strict=strict)
        return True

    def encode_mel(self, mel):
        return self.embedder(mel)

    def aggregate_embeddings(self, embeddings):
        stacked = torch.stack(embeddings, dim=0)
        return stacked.mean(dim=0)

    def forward(self, vowel_mels):
        embeddings = [self.encode_mel(mel) for mel in vowel_mels]
        return self.aggregate_embeddings(embeddings)
