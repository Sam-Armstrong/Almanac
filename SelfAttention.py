import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, encode_size = 240):
        super(SelfAttention, self).__init__()

        self.encode_size = encode_size
        self.query_encode = nn.Linear(encode_size, encode_size)
        self.key_encode = nn.Linear(encode_size, encode_size)
        self.value_encode = nn.Linear(encode_size, encode_size)

        self.softmax = nn.Softmax(dim = -1)

    def forward(self, x):
        queries = self.query_encode(x)
        keys = self.key_encode(x)
        values = self.value_encode(x)
        # queries shape: (batch_size, layer_size)
        
        x = self.softmax(torch.matmul(queries, keys.T) / math.sqrt(self.encode_size))
        x = torch.matmul(x, values)

        return x