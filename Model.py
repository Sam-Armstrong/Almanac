import torch
import torch.nn as nn
import math
from PretrainingModel import PretrainingModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Model(nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()

        self.model = PretrainingModel(440, 12, 10, 440).to(device)
        self.model.load_state_dict(torch.load('pretrained_model.pickle'))
        # for param in self.model.parameters():
        #     param.requires_grad = False

        self.model.fc_out = nn.Linear(440, 3, bias = False)
        self.model.softmax = nn.Softmax(dim = -1)
        # self.model.fc_out.requires_grad_ = True
        # self.model.softmax.requires_grad_ = True

        #self.model.fc_out.weight.data.normal_(0, math.sqrt(2 / 11)) # 2/ 440
        self.model.fc_out.weight.data.uniform_(-math.sqrt(1 / 200), math.sqrt(1 / 200))
    

    def forward(self, x, y):
        z = self.model(x, y)
        return z

    def evaluate(self):
        self.model.eval()