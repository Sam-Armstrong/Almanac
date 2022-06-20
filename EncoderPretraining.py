import torch
import torch.nn as nn
from EncoderPretrainingModel import EncoderModel
import matplotlib.pyplot as plt
import numpy as np
import pandas
import torch.optim as optim
import time
from Data import Data
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
match_stats = pandas.read_csv('MatchStats.csv')
time_series_data = pandas.read_csv('TimeSeriesData.csv')

class PreTrain():
    
    def __init__(self, model, means, stds):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.means = means.to(self.device)
        self.stds = stds.to(self.device)
        self.data = Data()

    def train(self, training_data):
        num_epochs = 400
        lr = 3e-5
        wd = lr
        batch_size = 1000
        warmup_steps = 50
        seq_len = 12
        n_features = 44
        n_out = 14

        start_time = time.time()
        plot_data = np.empty((num_epochs), dtype = float)

        X = torch.load('encoder_training_data.pt').to(self.device)
        y = torch.load('encoder_targets.pt').to(self.device)
        
        out_size = X.shape[2]
        X = X.reshape(X.shape[0] * X.shape[1], out_size)
        self.means = torch.mean(X, dim = 0)
        self.stds = torch.std(X, dim = 0)
        self.min = torch.min(X)
        self.max = torch.max(X)
        X = X.reshape(X.shape[0] // 12, 12, out_size)
        
        X -= self.means
        X /= self.stds

        train_data = []
        for i in range(len(X)):
            train_data.append([X[i], y[i]])

        print(X.shape[0])
        train_set, val_set = torch.utils.data.random_split(train_data, [45000, X.shape[0] - 45000]) # Splits the training data into a train set and a validation set

        train_dataloader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = 0)
        val_dataloader = torch.utils.data.DataLoader(val_set, batch_size = batch_size, shuffle = False, num_workers = 0)

        params = self.model.parameters()

        criterion = nn.MSELoss()
        optimizer = optim.Adam(params, lr = lr, weight_decay = wd)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [], gamma = 1e-1)
        
        pbar = tqdm(range(num_epochs))
        
        for epoch in pbar:
            epoch_start = time.time()
            
            # Learning rate warmup
            if epoch < warmup_steps:
                for g in optimizer.param_groups:
                    g['lr'] = lr * ((epoch + 1) / warmup_steps)

            train_loss = 0.0
            self.model.train()

            for batch_idx, (data, labels) in enumerate(train_dataloader):               
                data = data.float().to(self.device)
                labels = labels.float()
                avg_opp_data = labels[:, :, 14:].to(self.device)
                labels = labels[:, :, :14]
                labels = labels.to(self.device)

                if torch.sum(labels).item() != 0 and torch.sum(avg_opp_data).item() != 0:
                    scores = self.model(data, avg_opp_data) # Runs a forward pass of the model for all the data
                    loss = criterion(scores.float(), labels.float()).float() # Calculates the loss of the forward pass using the loss function
                    train_loss += loss

                    optimizer.zero_grad() # Resets the optimizer gradients to zero for each batch
                    loss.backward() # Backpropagates the network using the loss to calculate the local gradients

                    optimizer.step() # Updates the network weights and biases
                else:
                    print('skipping batch')

            valid_loss = 0.0
            self.model.eval()

            for batch_idx, (data, labels) in enumerate(val_dataloader):
                with torch.no_grad():
                    data = data.float().to(self.device)
                    labels = labels.float()
                    avg_opp_data = labels[:, :, 14:].to(self.device)
                    labels = labels[:, :, :14]
                    labels = labels.to(self.device)
                    
                    target = self.model(data, avg_opp_data)
                    loss = criterion(target, labels).float()
                    valid_loss = loss.item() * data.size(0)

            scheduler.step()

            plot_data[epoch] = valid_loss
            pbar.set_description(f"epoch {epoch + 1} validation loss {valid_loss:.5f}; lr {lr:e}; epoch_time {round(time.time() - epoch_start, 2):.5f}")

        print('Finished in %s seconds' % round(time.time() - start_time, 1))
        plt.plot(plot_data)
        plt.ylabel('Validation Loss')
        plt.xlabel('Epoch')
        plt.show()

        torch.save(self.model.state_dict(), 'pretrained_encoder_model.pickle')
        torch.save(self.means, 'means.pt')
        torch.save(self.stds, 'stds.pt')

if __name__ == '__main__':
    data = Data()
    model = EncoderModel(150, 12, 5, 250)
    means = torch.zeros((40))
    stds = torch.zeros((40))
    pretrain_data = data.pretrain_data
    p = PreTrain(model, means, stds)
    p.train(pretrain_data)