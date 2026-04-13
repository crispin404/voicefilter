import torch.nn as nn


class EmbeddingAdapter(nn.Module):
    def __init__(self, emb_dim, hidden_dim=None):
        super(EmbeddingAdapter, self).__init__()
        hidden_dim = hidden_dim or emb_dim
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emb_dim),
        )

    def forward(self, embedding):
        return embedding + self.net(embedding)
