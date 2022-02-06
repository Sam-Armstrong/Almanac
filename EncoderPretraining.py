import torch
import torch.nn as nn
import math
from EncoderPretrainingModel import EncoderModel
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
        print(len(training_data))

        num_epochs = 100
        lr = 5e-5 #5e-7 #1e-7 #3e-6 #8e-7 # Learning rate
        wd = 0 #1e-6 #3e-6 # Weight decay
        batch_size = 500
        warmup_steps = 2
        seq_len = 12
        n_features = 44
        n_out = 14

        start_time = time.time()
        plot_data = np.empty((num_epochs), dtype = float)

        X = torch.load('encoder_training_data.pt').to(self.device)
        y = torch.load('encoder_targets.pt').to(self.device)

        # X shape: (480, 100, 12, 44)
        # y shape: (480, 100, 12, 14)

        # # Split into data and labels
        # X = training_data.iloc[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 
        #                            28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44]].values
        # y = training_data.iloc[:, [45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57]].values

        self.means = torch.mean(X, dim = 0)
        self.stds = torch.std(X, dim = 0)
        
        X -= self.means
        X /= self.stds

        # X shape: (48004, 12, 44)
        # y shape: (48004, 12, 14)

        train_data = []
        for i in range(len(X)):
            train_data.append([X[i], y[i]])

        train_set, val_set = torch.utils.data.random_split(train_data, [45000, X.shape[0] - 45000]) # Splits the training data into a train set and a validation set

        train_dataloader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = 0)
        val_dataloader = torch.utils.data.DataLoader(val_set, batch_size = batch_size, shuffle = False, num_workers = 0)
        # X shape: (48004, 100, 12, 44)
        # y shape: (48004, 100, 12, 14)

        params = self.model.parameters()

        criterion = nn.MSELoss() #nn.L1Loss() #nn.MSELoss() #nn.CrossEntropyLoss()
        optimizer = optim.Adam(params, lr = lr, weight_decay = wd) # 5e-6   1e-8 #1e-3
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [], gamma = 1e-1) #3, 6, 10, 20, 30, 40, 50

        for epoch in range(num_epochs):
            # if epoch == 10: # Switch the loss function after x epochs #15
            #     criterion = nn.MSELoss() #nn.L1Loss() #nn.MSELoss()

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

                scores = self.model(data) # Runs a forward pass of the model for all the data
                #print(scores)
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
                    
                    target = self.model(data)
                    loss = criterion(target, labels).float()
                    valid_loss = loss.item() * data.size(0)

            #print(target)

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

        torch.save(self.model.state_dict(), 'pretrained_encoder_model.pickle')
        print('Saved model to .pickle file')
        torch.save(self.means, 'means.pt')
        torch.save(self.stds, 'stds.pt')
        print('Saved means and standard deviations')

if __name__ == '__main__':
    data = Data()
    model = EncoderModel(440, 12, 8, 440)
    means = torch.zeros((40))
    stds = torch.zeros((40))
    pretrain_data = data.pretrain_data
    p = PreTrain(model, means, stds)
    p.train(pretrain_data)