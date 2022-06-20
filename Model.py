import torch
import torch.nn as nn
from EncoderPretrainingModel import EncoderModel


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Model(nn.Module):
    
    def __init__(self, n_features, seq_len, n_layers, forward_expansion) -> None:
        super(Model, self).__init__()
        
        self.seq_len = seq_len
        self.n_features = n_features
        
        self.encoder_model = EncoderModel(n_features, seq_len, n_layers, forward_expansion).to(device)
        self.encoder_model.load_state_dict(torch.load('pretrained_encoder_model.pickle'))
        
        self.blocks = self.encoder_model.blocks
        self.expansion = self.encoder_model.expansion
        self.pe = self.encoder_model.position_embedding

        self.ln_out = nn.LayerNorm(n_features * 2)
        self.fc_out = nn.Linear(150 * 12 * 2, 3, bias = False)
        self.softmax = nn.Softmax(dim = -1)
    

    def forward(self, x, y):
        batch_size = x.shape[0]
        
        # Add the position embedding and expand the input data for both teams
        pos_embedding = self.pe(torch.arange(0, self.seq_len).reshape(1, self.seq_len).type(torch.LongTensor).to(device)).repeat(batch_size, 1, 1)
        x = torch.concat((pos_embedding, x), dim = -1)
        x = self.expansion(x)
        
        pos_embedding = self.pe(torch.arange(0, self.seq_len).reshape(1, self.seq_len).type(torch.LongTensor).to(device)).repeat(batch_size, 1, 1)
        y = torch.concat((pos_embedding, y), dim = -1)
        y = self.expansion(y)
        
        # Run the encoder on the input for team 1 and team 2
        x = self.blocks(x)
        y = self.blocks(y)

        # Concatenate the output for each team and run it through the final layers
        z = torch.concat((x, y), dim = -1)
        z = self.ln_out(z)
        z = z.reshape(z.shape[0], self.n_features * self.seq_len * 2)

        # Output, softmax, and scale
        z = self.fc_out(z)
        z = self.softmax(z)
        return z * 100

    def no_transformer_grad(self):
        for param in self.blocks.parameters():
            param.requires_grad = False
        for param in self.expansion.parameters():
            param.requires_grad = False
        for param in self.pe.parameters():
            param.requires_grad = False