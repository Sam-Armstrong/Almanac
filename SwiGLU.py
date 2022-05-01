import torch
import torch.nn as nn

class SwiGLU(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(SwiGLU, self).__init__()

        self.linear1 = nn.Linear(in_dim, out_dim, bias = False)
        self.linear2 = nn.Linear(in_dim, out_dim, bias = False)
        self.swish = nn.SiLU()

    def forward(self, x):

        y1 = self.linear1(x)
        y1 = self.swish(y1)
        y2 = self.linear2(x)
        out = torch.multiply(y1, y2)
        
        return out