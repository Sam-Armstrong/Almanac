import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, emsize, heads):
        super(SelfAttention, self).__init__()
        self.emsize = emsize
        self.heads = heads # Not currently used
        self.values = nn.Linear(emsize, emsize, bias = False)
        self.keys = nn.Linear(emsize, emsize, bias = False)
        self.queries = nn.Linear(emsize, emsize, bias = False)
        self.softmax = nn.Softmax(dim = -1)

    def forward(self, x, mask):
        ## Currently only uses one attention head
        # x shape: (batch_size, seq_len, emsize)

        batch_size = x.shape[0]
        seq_len = x.shape[1]
        
        values = self.values(x)
        keys = self.keys(x)
        queries = self.queries(x)
        # values/keys/queries shape: (batch_size, seq_len, emsize)

        # Flatten the batch
        values = values.reshape(values.shape[0] * values.shape[1], values.shape[2])
        keys = keys.reshape(keys.shape[0] * keys.shape[1], keys.shape[2])
        queries = queries.reshape(queries.shape[0] * queries.shape[1], queries.shape[2])
        # values/keys/queries shape: (batch_size * seq_len, emsize)

        similarities = torch.matmul(queries, keys.T) # similarities shape: (batch_size * seq_len, batch_size * seq_len)
        
        # Mask the input
        if mask is not None:
            similarities = similarities.masked_fill(mask == 0, float('-1e20'))

        print('Masked similarities', similarities)

        attention_weights = self.softmax(similarities / math.sqrt(self.emsize))

        output = torch.matmul(attention_weights, values) # output shape: (batch_size * seq_len, emsize)
        output = output.reshape(batch_size, seq_len, self.emsize) # output shape: (batch_size, seq_len, emsize)

        return output