import torch
import torch.nn as nn
import math

class Model(nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()

        self.dropout = nn.Dropout(p = 0.1)
        self.gelu = nn.GELU()
        self.softmax = nn.Softmax(dim = -1)
        
        self.fc1 = nn.Linear(29, 580, bias = False) # , bias = False ??
        self.fc2 = nn.Linear(580, 1160, bias = True)
        self.fc3 = nn.Linear(1160, 580, bias = True)
        self.fc4 = nn.Linear(580, 1160, bias = True)
        self.fc5 = nn.Linear(1160, 580, bias = True)
        self.fc6 = nn.Linear(580, 1160, bias = True)
        self.fc7 = nn.Linear(1160, 580, bias = True)
        self.fc8 = nn.Linear(580, 1160, bias = True)
        self.fc9 = nn.Linear(1160, 580, bias = True)
        self.fc10 = nn.Linear(580, 3, bias = False)

        # Kaiming (He) weight initialization for fully connected layers
        self.fc1.weight.data.normal_(0, math.sqrt(2 / 580))
        self.fc2.weight.data.normal_(0, math.sqrt(2 / 1740)) # 2 / math.sqrt(580)
        self.fc3.weight.data.normal_(0, math.sqrt(2 / 1740))
        self.fc4.weight.data.normal_(0, math.sqrt(2 / 1740))
        self.fc5.weight.data.normal_(0, math.sqrt(2 / 1740))
        self.fc6.weight.data.normal_(0, math.sqrt(2 / 1740))
        self.fc7.weight.data.normal_(0, math.sqrt(2 / 1740))
        self.fc8.weight.data.normal_(0, math.sqrt(2 / 1740))
        self.fc9.weight.data.normal_(0, math.sqrt(2 / 1740))
        self.fc10.weight.data.normal_(0, math.sqrt(2 / 580)) # 2/3

        # self.div_param = torch.nn.Parameter(torch.tensor(200.0))

        # self.fc1.bias.data.zero_()
        self.fc2.bias.data.zero_()
        self.fc3.bias.data.zero_()
        self.fc4.bias.data.zero_()
        self.fc5.bias.data.zero_()
        self.fc6.bias.data.zero_()
        self.fc7.bias.data.zero_()
        self.fc8.bias.data.zero_()
        self.fc9.bias.data.zero_()
        # self.fc10.bias.data.zero_()

        self.ln = nn.LayerNorm((580))
        self.ln1 = nn.LayerNorm((580))
        self.ln2 = nn.LayerNorm((580))
        self.ln3 = nn.LayerNorm((580))
        self.ln4 = nn.LayerNorm((580))
        self.ln5 = nn.LayerNorm((580))
        self.ln6 = nn.LayerNorm((580))
        self.ln7 = nn.LayerNorm((580))
        self.ln8 = nn.LayerNorm((580))

    def forward(self, x):
        x = self.fc1(x)
        #x = self.ln(x)
        x = self.dropout(x)

        #res = x.clone()
        x = self.fc2(x)
        x = self.gelu(x)
        x = self.fc3(x)
        #x += res
        x = self.ln2(x)
        x = self.dropout(x)

        #res = x.clone()
        x = self.fc4(x)
        x = self.gelu(x)
        x = self.fc5(x)
        #x += res
        x = self.ln3(x)
        x = self.dropout(x)

        #res = x.clone()
        x = self.fc6(x)
        x = self.gelu(x)
        x = self.fc7(x)
        #x += res
        x = self.ln4(x)
        x = self.dropout(x)

        #res = x.clone()
        x = self.fc8(x)
        x = self.gelu(x)
        x = self.fc9(x)
        #x += res
        x = self.ln5(x)

        x = self.fc10(x)

        x = x / math.sqrt(580) #120 #580 #math.sqrt(580) # Scale to reduce vanishing gradient problem
        x = self.softmax(x)

        return x