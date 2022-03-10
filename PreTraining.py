import torch
import torch.nn as nn
import math
from PretrainingModel import PretrainingModel
import matplotlib.pyplot as plt
import numpy as np
import pandas
import torch.optim as optim
import time
from Data import Data
import einops
from torch import Tensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
match_stats = pandas.read_csv('MatchStats.csv')
time_series_data = pandas.read_csv('TimeSeriesData.csv')

def generate_mask(seq_len, batch_size = 100):
    mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal = 1)
    #mask = mask.repeat(batch_size) ## Check if this is correct
    return mask

class PreTrain():
    def __init__(self, model, means, stds):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.means = means.to(self.device)
        self.stds = stds.to(self.device)
        self.data = Data()

    def train(self, training_data):
        num_epochs = 100
        lr = 2e-5 #1e-4  #2e-4 #5e-7 #1e-7 #3e-6 #8e-7 # Learning rate
        wd = 0 #1e-6 #3e-6 # Weight decay
        batch_size = 250
        warmup_steps = 30
        seq_len = 12
        n_features = 44
        n_out = 14

        start_time = time.time()
        plot_data = np.empty((num_epochs - 1), dtype = float)

        X = torch.load('pretraining_data.pt').to(self.device)
        y = torch.load('pretraining_targets.pt').to(self.device)

        # X shape: (18911, 12, 88)
        # y shape: (18911, 28)

        self.means = torch.mean(X, dim = 0)
        self.stds = torch.std(X, dim = 0)
        self.min = torch.min(X)
        self.max = torch.max(X)
        
        X -= self.means
        X /= self.stds
        #X -= self.min
        #X /= self.max

        train_data = []
        for i in range(len(X)):
            train_data.append([X[i], y[i]])
        print(len(train_data))
        train_set, val_set = torch.utils.data.random_split(train_data, [17000, X.shape[0] - 17000]) # Splits the training data into a train set and a validation set

        train_dataloader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = 0)
        val_dataloader = torch.utils.data.DataLoader(val_set, batch_size = batch_size, shuffle = False, num_workers = 0)

        params = self.model.parameters()

        criterion = nn.MSELoss() #nn.L1Loss() #nn.MSELoss() #nn.CrossEntropyLoss()
        optimizer = optim.Adam(params, lr = lr, weight_decay = wd) # 5e-6   1e-8 #1e-3
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [], gamma = 1e-1) #3, 6, 10, 20, 30, 40, 50

        for epoch in range(num_epochs):
            # Learning rate warmup
            if epoch < warmup_steps:
                for g in optimizer.param_groups:
                    g['lr'] = lr * ((epoch + 1) / warmup_steps)

            epoch_start = time.time()

            print('Epoch: ', epoch)
            train_loss = 0.0
            self.model.train()

            for batch_idx, (data, labels) in enumerate(train_dataloader):
                data = data.float().to(self.device)
                labels = labels.to(self.device)
                #labels = einops.repeat(labels, 'b n -> b s n', s = seq_len)
                #print(data[0, :, 5])

                team1 = data[:, :, :44] # shape: (500, 12, 44)
                team2 = data[:, :, 44:]
                scores = self.model(team1, team2) # Runs a forward pass of the model for all the data
                loss = criterion(scores.float(), labels.float()).float() # Calculates the loss of the forward pass using the loss function
                train_loss += loss

                optimizer.zero_grad() # Resets the optimizer gradients to zero for each batch
                loss.backward() # Backpropagates the network using the loss to calculate the local gradients
                optimizer.step() # Updates the network weights and biases

            valid_loss = 0.0
            self.model.eval()

            for batch_idx, (data, labels) in enumerate(val_dataloader):
                with torch.no_grad():
                    data = data.float().to(self.device)
                    labels = labels.to(self.device)
                    #labels = einops.repeat(labels, 'b n -> b s n', s = seq_len)

                    team1 = data[:, :, :44] # shape: (500, 12, 44)
                    team2 = data[:, :, 44:]
                    
                    target = self.model(team1, team2)
                    #print(target[0])
                    loss = criterion(target, labels).float()
                    valid_loss += loss.item() * data.size(0)

            scheduler.step()

            # valid_accuracy = check_accuracy(val_dataloader)
            # print(valid_accuracy, '% Validation Accuracy')
            print('Validation Loss: ', valid_loss)
            if epoch != 0:
                plot_data[epoch - 1] = valid_loss
            print('Epoch time: %s seconds' % round(time.time() - epoch_start, 2))

        print('Finished in %s seconds' % round(time.time() - start_time, 1))
        plt.plot(plot_data)
        plt.ylabel('Validation Loss')
        plt.xlabel('Epoch')
        plt.show()

        torch.save(self.model.state_dict(), 'pretrained_model.pickle')
        print('Saved model to .pickle file')
        torch.save(self.means, 'pre_means.pt')
        torch.save(self.stds, 'pre_stds.pt')
        print('Saved means and standard deviations')

if __name__ == '__main__':
    data = Data()
    model = PretrainingModel(44, 12, 5, 220)
    means = torch.zeros((40))
    stds = torch.zeros((40))
    pretrain_data = data.pretrain_data
    p = PreTrain(model, means, stds)
    p.train(pretrain_data)