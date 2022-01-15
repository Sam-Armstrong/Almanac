"""
Author: Sam Armstrong
Date: 2021
Description: Predictor model using self-attention layers
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
from Data import Data
import matplotlib.pyplot as plt
import numpy as np
import math

from SelfAttention import SelfAttention

def MAPELoss(output, target):
    loss = torch.mean(torch.abs((target - output) / target))
    return loss

class Model(nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()

        self.dropout = nn.Dropout(p = 0.2)
        self.gelu = nn.GELU()
        self.softmax = nn.Softmax(dim = -1)
        
        self.fc1 = nn.Linear(29, 580) # , bias = False ??
        self.fc2 = nn.Linear(580, 1160)
        self.fc3 = nn.Linear(1160, 580)
        self.fc4 = nn.Linear(580, 1160)
        self.fc5 = nn.Linear(1160, 580)
        self.fc6 = nn.Linear(580, 1160)
        self.fc7 = nn.Linear(1160, 580)
        self.fc8 = nn.Linear(580, 1160)
        self.fc9 = nn.Linear(1160, 580)
        self.fc10 = nn.Linear(580, 3)

        # Kaiming (He) weight initialization for fully connected layers
        self.fc1.weight.data.normal_(0, 2 / math.sqrt(580))
        self.fc1.bias.data.zero_()
        self.fc2.weight.data.normal_(0, 2 / math.sqrt(580))
        self.fc2.bias.data.zero_()
        self.fc3.weight.data.normal_(0, 2 / math.sqrt(580))
        self.fc3.bias.data.zero_()
        self.fc4.weight.data.normal_(0, 2 / math.sqrt(580))
        self.fc4.bias.data.zero_()
        self.fc5.weight.data.normal_(0, 2 / math.sqrt(580))
        self.fc5.bias.data.zero_()
        self.fc6.weight.data.normal_(0, 2 / math.sqrt(580))
        self.fc6.bias.data.zero_()
        self.fc7.weight.data.normal_(0, 2 / math.sqrt(580))
        self.fc7.bias.data.zero_()
        self.fc8.weight.data.normal_(0, 2 / math.sqrt(580))
        self.fc8.bias.data.zero_()
        self.fc9.weight.data.normal_(0, 2 / math.sqrt(580))
        self.fc9.bias.data.zero_()
        self.fc10.weight.data.normal_(0, 2 / math.sqrt(3))
        self.fc10.bias.data.zero_()

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
        x = self.ln(x)

        #res = x.clone()
        x = self.fc2(x)
        x = self.gelu(x)
        x = self.fc3(x)
        #x += res
        x = self.ln2(x)
        #x = self.dropout(x)

        #res = x.clone()
        x = self.fc4(x)
        x = self.gelu(x)
        x = self.fc5(x)
        #x += res
        x = self.ln3(x)
        #x = self.dropout(x)

        #res = x.clone()
        x = self.fc6(x)
        x = self.gelu(x)
        x = self.fc7(x)
        #x += res
        x = self.ln4(x)
        #x = self.dropout(x)

        #res = x.clone()
        x = self.fc8(x)
        x = self.gelu(x)
        x = self.fc9(x)
        #x += res
        x = self.ln5(x)

        x = self.fc10(x)

        x = x / math.sqrt(580) # Reduce vanishing gradient problem
        x = self.softmax(x)

        return x


class Predictor:
    def __init__(self, model_name = 'model.pickle'):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.means = torch.zeros((40))
        self.stds = torch.zeros((40))
        self.model = Model().to(device = self.device)
        try:
            self.model.load_state_dict(torch.load(model_name))
            self.means = torch.load('means.pt')
            self.stds = torch.load('stds.pt')
        except:
            print('No previously trained model found')
            pass

        self.means = self.means.to(self.device)
        self.stds = self.stds.to(self.device)

    # Method for making predictions using the model
    def predict(self, prediction_data): # prediction_data2 ??
        self.model.eval()
        prediction_data = torch.from_numpy(prediction_data).float().to(self.device)
        prediction_data -= self.means
        prediction_data /= self.stds
        prediction = self.model(prediction_data)
        return prediction

    # Method for training the model using a given set of data
    def train(self, training_data):

        self.model = Model().to(device = self.device)

        num_epochs = 30
        lr = 8e-7 #3e-6 #8e-7 # Learning rate
        wd = 0 # Weight decay
        batch_size = 500

        start_time = time.time()
        plot_data = np.empty((num_epochs), dtype = float)

        # The data is split into training data and labels later on
        X = training_data.iloc[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]].values
        y = training_data.iloc[:, [30, 31, 32]].values

        X = torch.tensor(X).float()
        y = torch.tensor(y).float()

        self.means = torch.mean(X, dim = 0)
        self.stds = torch.std(X, dim = 0)
        
        X -= self.means
        X /= self.stds

        train_data = []
        for i in range(len(X)):
            train_data.append([X[i], y[i]])

        train_set, val_set = torch.utils.data.random_split(train_data, [45000, X.shape[0] - 45000]) # Splits the training data into a train set and a validation set

        train_dataloader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = 4)
        val_dataloader = torch.utils.data.DataLoader(val_set, batch_size = batch_size, shuffle = False, num_workers = 4)

        params = []
        params += self.model.parameters()

        criterion = nn.CrossEntropyLoss() #nn.KLDivLoss() #nn.L1Loss() #nn.MSELoss() #nn.CrossEntropyLoss()
        optimizer = optim.Adam(params, lr = lr, weight_decay = wd) # 5e-6   1e-8 #1e-3
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [], gamma = 1e-2) #3, 6, 10, 20, 30, 40, 50

        # Checks the performance of the model on the test set
        def check_accuracy(dataset):
            num_correct = 0
            num_samples = 0
            self.model.eval()

            with torch.no_grad():
                for batch_idx, (data, labels) in enumerate(dataset):
                    data = data.float().to(device = self.device)
                    labels = labels.to(device = self.device)

                    scores = self.model(data)
                    _, predictions = scores.max(1)
                    labels = torch.max(labels, dim = 1)[1]
                    num_correct += (predictions == labels).sum()
                    num_samples += predictions.size(0)
            
            return (num_correct * 100 / num_samples).item()


        for epoch in range(num_epochs):
            # if epoch == 20: # Switch the loss function after x epochs #15
            #     criterion = nn.MSELoss() #nn.L1Loss() #nn.MSELoss()

            # Learning rate warmup
            if epoch < 20:
                for g in optimizer.param_groups:
                    g['lr'] = lr * ((epoch + 1) / 20)

            epoch_start = time.time()

            print('Epoch: ', epoch)
            train_loss = 0.0
            self.model.train()

            for batch_idx, (data, labels) in enumerate(train_dataloader):                
                data = data.float().to(device = self.device)
                labels = labels.to(device = self.device)

                scores = self.model(data) # Runs a forward pass of the model for all the data
                #print(scores)
                loss = criterion(scores.float(), labels.float()).float() # Calculates the loss of the forward pass using the loss function
                train_loss += loss

                # print(scores.float())
                # print(labels.float())

                #self.model.zero_grad() ###
                optimizer.zero_grad() # Resets the optimizer gradients to zero for each batch
                loss.backward() # Backpropagates the network using the loss to calculate the local gradients

                optimizer.step() # Updates the network weights and biases

            valid_loss = 0.0
            self.model.eval()

            for batch_idx, (data, labels) in enumerate(val_dataloader):
                with torch.no_grad():
                    data = data.float().to(device = self.device)
                    labels = labels.to(device = self.device)
                    
                    target = self.model(data)
                    loss = criterion(target, labels).float()
                    valid_loss = loss.item() * data.size(0)

            scheduler.step()

            # valid_accuracy = check_accuracy(val_dataloader)
            # print(valid_accuracy, '% Validation Accuracy')
            print('Validation Loss: ', valid_loss)

            plot_data[epoch] = valid_loss
            print('Epoch time: %s seconds' % round(time.time() - epoch_start, 2))

        print('Finished in %s seconds' % round(time.time() - start_time, 1))
        plt.plot(plot_data)
        plt.ylabel('Validation Loss')
        plt.xlabel('Epoch')
        plt.show()

        torch.save(self.model.state_dict(), 'model.pickle')
        print('Saved model to .pickle file')
        torch.save(self.means, 'means.pt')
        torch.save(self.stds, 'stds.pt')
        print('Saved means and standard deviations')


if __name__ == '__main__':
    predictor = Predictor()
    data = Data()
    training_data = data.training_data
    predictor.train(training_data)