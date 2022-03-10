import torch
import torch.nn as nn

class adaSoft(nn.Module):
    def __init__(self, weight_init = 1) -> None:
        super(adaSoft, self).__init__()
        self.softmax = nn.Softmax(dim = -1)
        initial_scaler = torch.scalar_tensor(weight_init) #torch.Tensor((weight_init))
        self.scaler = nn.Parameter(data = initial_scaler, requires_grad = True)

    def forward(self, x):
        x = self.softmax(x)
        x *= self.scaler
        return x