import torch
import torch.nn as nn
import math
from PretrainingModel import PretrainingModel
from adaSoft import adaSoft

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Model(nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()

        self.model = PretrainingModel(440, 12, 5, 880).to(device)
        self.model.load_state_dict(torch.load('pretrained_model.pickle'))
        # for param in self.model.parameters():
        #     param.requires_grad = False

        self.model.fc_out = nn.Linear(440, 3, bias = False)
        #self.softmax = nn.Softmax(dim = -1)
        # self.model.fc_out.requires_grad_ = True
        # self.model.softmax.requires_grad_ = True
        self.softmax = nn.Softmax(dim = -1) #adaSoft(weight_init = 1 / math.sqrt(440)) #1

        self.model.fc_out.weight.data.normal_(0, math.sqrt(2 / 440)) # 2/ 440
        #self.model.fc_out.weight.data.uniform_(-math.sqrt(1 / 44), math.sqrt(1 / 44))
        #self.model.fc_out.weight.data.uniform_(-math.sqrt(2 / 440), math.sqrt(2 / 440))
    

    def forward(self, x, y):
        z = self.model(x, y)
        z = self.softmax(z) # / math.sqrt(1 / 44))
        ## Could make the softmax scaler a learnable parameter that is trained through backpropagation?
        return z

    def evaluate(self):
        self.model.eval()