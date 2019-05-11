import torch
import torch.nn as nn
import torch.nn.functional as f


class Model(nn.Module):
    def __init__(self, num_emb:int, emb_dim:int, hidden_dim:int, num_class:int):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(num_emb, emb_dim)
        self.hidden = nn.Linear(emb_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, num_class)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.hidden(x)
        x = f.relu(x)
        x = self.decoder(x)
        return x