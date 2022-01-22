import torch
import torch.nn as nn
import math
from PretrainingModel import Model

class FinetunedModel(nn.Module):
    def __init__(self) -> None:
        super(FinetunedModel, self).__init__()
        
        self.pretrained = Model().to(torch.device('cuda'))
        self.pretrained.load_state_dict(torch.load('pretrained_model.pickle'))
        # New layers
        self.pretrained.fc10 = nn.Linear(580, 3, bias = False)
        self.pretrained.softmax = nn.Softmax(dim = -1)
        self.pretrained.fc10.weight.data.normal_(0, math.sqrt(2 / 580))

        # Freeze pretrained weights
        # for param in self.model.parameters():
        #     param.requires_grad = False

    def forward(self, x):
        x = self.pretrained(x)
        return x