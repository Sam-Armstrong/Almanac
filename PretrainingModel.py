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
        for param in self.encoder_model.parameters():
            param.requires_grad = False
        #self.encoder_model.fc_out = nn.Linear(440, 220)

        self.blocks = self.encoder_model.blocks
        self.expansion = self.encoder_model.expansion
        self.pe = self.encoder_model.positional_encoding
        for block in self.blocks:
            for param in block.parameters():
                param.requires_grad = False
        for param in self.pe.parameters():
            param.requires_grad = False
        for param in self.expansion.parameters():
            param.requires_grad = False

        self.ln_out = nn.LayerNorm(440) # Normalizes the transerable output, in order to allow more effective learning of finetuning tasks (only needs to be included if using multiple linear layers prior to softmax)
        self.fc_out = nn.Linear(440, 28, bias = False)

        #self.fc1 = nn.Linear(88, 440, bias = False) # , bias = False ??
        self.fc2 = nn.Linear(880, 440, bias = False)
        self.fc3 = nn.Linear(440, 440, bias = False)
        self.fc4 = nn.Linear(440, 440, bias = False)
        self.fc5 = nn.Linear(440, 440, bias = False)
        self.fc6 = nn.Linear(440, 440, bias = False)
        self.fc7 = nn.Linear(440, 440, bias = False)
        self.fc8 = nn.Linear(440, 440, bias = False)
        self.fc9 = nn.Linear(440, 440, bias = False)

        # Kaiming (He) weight initialization for fully connected layers
        #self.fc1.weight.data.normal_(0, math.sqrt(2 / 88))
        # self.fc2.weight.data.normal_(0, math.sqrt(2 / 880)) # 2 / math.sqrt(580)
        # self.fc3.weight.data.normal_(0, math.sqrt(2 / 440))
        # self.fc4.weight.data.normal_(0, math.sqrt(2 / 440))
        # self.fc5.weight.data.normal_(0, math.sqrt(2 / 440))
        # self.fc6.weight.data.normal_(0, math.sqrt(2 / 440))
        # self.fc7.weight.data.normal_(0, math.sqrt(2 / 440))
        # self.fc8.weight.data.normal_(0, math.sqrt(2 / 440))
        # self.fc9.weight.data.normal_(0, math.sqrt(2 / 440))
        # self.fc_out.weight.data.normal_(0, math.sqrt(2 / 440)) # 2/3

        # self.div_param = torch.nn.Parameter(torch.tensor(200.0))

        # self.fc1.bias.data.zero_()
        # self.fc2.bias.data.zero_()
        # self.fc3.bias.data.zero_()
        # self.fc4.bias.data.zero_()
        # self.fc5.bias.data.zero_()
        # self.fc6.bias.data.zero_()
        # self.fc7.bias.data.zero_()
        # self.fc8.bias.data.zero_()
        # self.fc9.bias.data.zero_()
        # self.fc10.bias.data.zero_()

        self.ln = nn.LayerNorm((440))
        self.ln1 = nn.LayerNorm((440))
        self.ln2 = nn.LayerNorm((440))
        self.ln3 = nn.LayerNorm((440))
        self.ln4 = nn.LayerNorm((440))
        self.ln5 = nn.LayerNorm((440))
        self.ln6 = nn.LayerNorm((440))
        self.ln7 = nn.LayerNorm((440))
        self.ln8 = nn.LayerNorm((440))

        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p = 0.2)

    def forward(self, x, y):
        # #x = self.expansion(x)
        # x = self.pe(x)
        # #y = self.expansion(y)
        # y = self.pe(y)
        
        # # Run the encoder on the input for team 1 and team 2
        # for block in self.blocks:
        #     x = block(x)

        # for block in self.blocks:
        #     y = block(y)

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

        ### Implement same network here that is used in non-transformer model

        # z = self.fc1(z)
        # #res = z.clone()
        # z = self.gelu(z)
        # z = self.fc2(z)
        # z = self.gelu(z) # Only applies a single GELU non-linearity in the fully-connected block
        # z = self.dropout(z)

        # z = self.fc3(z)
        # z = self.gelu(z)
        # #z += res
        # z = self.fc4(z)
        # #z = self.gelu(z)
        # z = self.dropout(z)
        # z = self.ln_out(z)

        # z = self.fc_out(z)

        z = self.fc2(z)
        z = self.gelu(z)
        z = self.fc3(z)
        z = self.ln2(z)
        z = self.dropout(z)

        z = self.fc4(z)
        z = self.gelu(z)
        z = self.fc5(z)
        z = self.ln3(z)
        z = self.dropout(z)

        z = self.fc6(z)
        z = self.gelu(z)
        z = self.fc7(z)
        z = self.ln4(z)
        z = self.dropout(z)

        # #res = z.clone()
        # z = self.dropout(z)
        # z = self.fc6(z)
        # z = self.gelu(z)
        # z = self.fc7(z)
        # z = self.ln5(z)

        z = self.fc_out(z)

        return z
