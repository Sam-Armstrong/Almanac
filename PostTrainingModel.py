import torch
import torch.nn as nn
import math
from Model import Model
from adaSoft import adaSoft

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PostTrainingModel(nn.Module):
    def __init__(self) -> None:
        super(PostTrainingModel, self).__init__()

        self.model = torch.load('best_model.pt').to(device)
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.softmax = adaSoft(weight_init = 10)

    def forward(self, x, y):
        z = self.model(x, y)
        return z