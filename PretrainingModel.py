import torch
import torch.nn as nn
from SelfAttention import SelfAttention
from EncoderPretrainingModel import EncoderModel, TransformerBlock
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PretrainingModel(nn.Module):
    def __init__(self, n_features, seq_len, n_layers, forward_expansion):
        super(PretrainingModel, self).__init__()

        self.seq_len = seq_len

        self.encoder_model = EncoderModel(n_features, seq_len, n_layers, forward_expansion).to(device)
        self.encoder_model.load_state_dict(torch.load('pretrained_encoder_model.pickle'))
        self.blocks = self.encoder_model.blocks
        self.expansion = self.encoder_model.expansion
        self.pe = self.encoder_model.postional_encoding

        # No attention mask is used
        for block in self.blocks:
            block.pretrain = False #lambda x: None #torch.ones(seq_len, seq_len).to(device)

        self.fc1 = nn.Linear(880, 440, bias = False)
        self.fc2 = nn.Linear(440, 440, bias = False)
        self.fc3 = nn.Linear(440, 440, bias = False)
        self.fc4 = nn.Linear(440, 440, bias = False)
        #self.ln_out = nn.LayerNorm(440) # Normalizes the transerable output, in order to allow more effective learning of finetuning tasks (only needs to be included if using multiple linear layers prior to softmax)
        self.fc_out = nn.Linear(440, 28, bias = False)

        self.fc_out.weight.data.uniform_(-math.sqrt(2 / 440), math.sqrt(2 / 440))
        self.fc1.weight.data.uniform_(-math.sqrt(2 / 440), math.sqrt(2 / 440))
        self.fc2.weight.data.uniform_(-math.sqrt(2 / 440), math.sqrt(2 / 440))
        self.fc3.weight.data.uniform_(-math.sqrt(2 / 440), math.sqrt(2 / 440))
        self.fc4.weight.data.uniform_(-math.sqrt(2 / 440), math.sqrt(2 / 440))

        self.softmax = lambda x: x
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p = 0.1)

    def forward(self, x, y):
        x = self.expansion(x)
        x = self.pe(x)
        y = self.expansion(y)
        y = self.pe(y)
        
        # Run the encoder on the input for team 1 and team 2
        for block in self.blocks:
            x = block(x)

        for block in self.blocks:
            y = block(y)

        # Concatenate the output for each team and run it through the final layers
        z = torch.concat((x, y), dim = -1)
        #z = z.reshape(z.shape[0], self.seq_len * 88)
        # z shape: (batch_size, seq_len, n_features * 2)

        #z = z[:, -1, :]
        # print(z.shape)
        # print(z)
        #z = torch.mean(z, dim = 1)

        z = self.fc1(z)
        res = z.clone()
        z = self.fc2(z)
        z = self.gelu(z) # Only applies a single GELU non-linearity in the fully-connected block
        z = self.dropout(z)
        #z = self.ln_out(z)

        z = self.fc3(z)
        z += res
        z = self.fc4(z)
        z = self.dropout(z)

        z = self.fc_out(z)
        z = self.softmax(z)

        return z
