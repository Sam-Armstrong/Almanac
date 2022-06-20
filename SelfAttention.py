import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, emsize, heads = 1):
        super(SelfAttention, self).__init__()
        self.emsize = emsize
        self.heads = heads
        self.head_dim = emsize // self.heads
        self.values = nn.Linear(emsize, emsize, bias = False)
        self.keys = nn.Linear(emsize, emsize, bias = False)
        self.queries = nn.Linear(emsize, emsize, bias = False)
        self.softmax = nn.Softmax(dim = -1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x, mask):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        
        values = self.values(x)
        keys = self.keys(x)
        queries = self.queries(x)

        values = values.reshape(batch_size, seq_len, self.heads, self.head_dim)
        keys = keys.reshape(batch_size, seq_len, self.heads, self.head_dim)
        queries = queries.reshape(batch_size, seq_len, self.heads, self.head_dim)

        similarities = torch.einsum('nqhd,nkhd->nhqk', [queries, keys])

        if mask is not None:
            mask = mask.repeat(batch_size, self.heads, 1, 1).to(self.device)
            similarities = torch.multiply(similarities, mask)

        attention_weights = self.softmax(similarities / math.sqrt(self.emsize))
        output = torch.einsum('nhql,nlhd->nqhd', [attention_weights, values]).reshape(batch_size, seq_len, self.emsize)

        return output