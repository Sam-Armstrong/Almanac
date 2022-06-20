import torch
import torch.nn as nn
from SelfAttention import SelfAttention

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EncoderModel(nn.Module):
    
    def __init__(self, n_features, seq_len, n_layers, forward_expansion):
        super(EncoderModel, self).__init__()

        self.seq_len = seq_len
        self.expansion = nn.Linear(44 * 2, n_features, bias = False)
        self.position_embedding = nn.Embedding(seq_len, 44)
        self.blocks = nn.Sequential(*[TransformerBlock(n_features, seq_len, forward_expansion).to(device) for _ in range(n_layers)])
        self.opp_expansion = nn.Linear(14, n_features, bias = False)
        self.fc_out = nn.Linear(n_features * 2, 14, bias = False)
        self.ln_out = nn.LayerNorm(n_features * 2)

    def forward(self, x, y):
        batch_size = x.shape[0]
        
        pos_embedding = self.position_embedding(torch.arange(0, self.seq_len).reshape(1, self.seq_len).type(torch.LongTensor).to(device)).repeat(batch_size, 1, 1)
        x = torch.concat((pos_embedding, x), dim = -1)
        x = self.expansion(x)
        y = self.opp_expansion(y)
        
        # Run the encoder on the input for team 1
        x = self.blocks(x)

        z = torch.concat((x, y), dim = -1)
        z = self.ln_out(z)
        return self.fc_out(z)



class TransformerBlock(nn.Module):
    
    def __init__(self, n_features, seq_len, forward_expansion):
        super(TransformerBlock, self).__init__()

        self.seq_len = seq_len
        self.att = SelfAttention(n_features, heads = 5).to(device)
        self.ln1 = nn.LayerNorm(n_features)
        self.ln2 = nn.LayerNorm(n_features)
        self.fc1 = nn.Linear(n_features, forward_expansion, bias = False)
        self.fc2 = nn.Linear(forward_expansion, n_features, bias = False)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p = 0.1)
        self.pretrain = True

    def forward(self, x):

        res = x.clone()
        x = self.ln1(x)
        x = self.att(x, None)
        x = self.dropout(x)
        x += res

        res = x.clone()
        x = self.ln2(x)
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x += res

        return x