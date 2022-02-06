import torch
import torch.nn as nn
from SelfAttention import SelfAttention
from torch import Tensor
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_mask(seq_len, batch_size = 100):
    mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal = 1).to(device)
    #mask = mask.repeat(batch_size) ## Check if this is correct
    return mask

class EncoderModel(nn.Module):
    def __init__(self, n_features, seq_len, n_layers, forward_expansion):
        super(EncoderModel, self).__init__()
        self.expansion = nn.Linear(44, n_features)
        self.postional_encoding = PositionalEncoding(n_features, max_len = 12).to(device)
        self.blocks = [TransformerBlock(n_features, seq_len, forward_expansion).to(device) for _ in range(n_layers)]
        self.fc_out = nn.Linear(n_features, 14)

    def forward(self, x):
        x = self.expansion(x)
        x = self.postional_encoding(x)
        
        # Run the encoder on the input for team 1 and team 2
        for block in self.blocks:
            x = block(x)

        z = self.fc_out(x)
        return z

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p = dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        self.pe = torch.zeros(max_len, 1, embed_dim).to(device)
        self.pe[:, 0, 0::2] = torch.sin(position * div_term)
        self.pe[:, 0, 1::2] = torch.cos(position * div_term)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
            x shape: (batch_size, seq_len, n_features)
        """
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        n_features = x.shape[2]
        x = x.reshape(seq_len, batch_size, n_features)
        x = x + self.pe[:x.size(0)]
        x = x.reshape(batch_size, seq_len, n_features)
        return self.dropout(x)

class TransformerBlock(nn.Module):
    def __init__(self, n_features, seq_len, forward_expansion):
        super(TransformerBlock, self).__init__()

        self.seq_len = seq_len
        self.att = SelfAttention(n_features, 1).to(device)
        self.ln1 = nn.LayerNorm(n_features)
        self.ln2 = nn.LayerNorm(n_features)
        self.fc1 = nn.Linear(n_features, forward_expansion, bias = False)
        self.fc2 = nn.Linear(forward_expansion, n_features, bias = False)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p = 0.1)
        self.pretrain = True

    def forward(self, x):
        if self.pretrain:
            mask = generate_mask(self.seq_len)
        else:
            mask = None

        res = x.clone()
        x = self.att(x, mask)
        x = self.dropout(x)
        x += res
        x = self.ln1(x)

        res = x.clone()
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x += res
        x = self.ln2(x)

        return x