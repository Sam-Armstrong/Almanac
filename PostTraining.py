import torch
import torch.nn as nn
import math
from PostTrainingModel import PostTrainingModel
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

class Train():
    def __init__(self, model, means, stds):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.means = means.to(self.device)
        self.stds = stds.to(self.device)
        self.data = Data()

    def test(self):
        # predictors = []
        # predictors.append(Predictor(model_name = 'trained_model.pickle', eval = True))
        
        #predictor = Predictor()
        data = Data()

        odds_df = pandas.read_csv('TestOdds1.csv')
        i = 0
        returns = 0
        t_est = 0
        num_correct = 0
        num_matches = 0

        # model = Model().to(device)
        # model.load_state_dict(torch.load('trained_model.pickle'))
        # model.eval()
        #model = Model
        model = self.model
        model.eval()

        means = torch.load('trained_means.pt').to(device)
        stds = torch.load('trained_stds.pt').to(device)
        means = means.reshape(1, means.shape[0], means.shape[1])
        stds = stds.reshape(1, stds.shape[0], stds.shape[1])

        for index, row in odds_df.iterrows():
            try:
                t_chance_win = 0
                t_chance_draw = 0
                t_chance_loss = 0

                #for predictor in predictors:
                date = row[0]
                team1 = row[1]
                team2 = row[2]
                result = row[3]
                odds1 = row[4]
                odds2 = row[5]
                odds3 = row[6]

                prediction_data1 = data.findTimeSeries(team1, team2, date)
                prediction_data2 = data.findTimeSeries(team2, team1, date)

                # if index == 1:
                #     print(prediction_data1)

                model.eval()
                prediction_data1 = prediction_data1.to(device)
                prediction_data1 -= means
                prediction_data1 /= stds
                prediction1 = model.forward(prediction_data1[:, :, :44], prediction_data1[:, :, 44:])
                prediction1 = torch.mean(prediction1, dim = 1)

                prediction_data2 = prediction_data2.to(device)
                prediction_data2 -= means
                prediction_data2 /= stds
                prediction2 = model.forward(prediction_data2[:, :, :44], prediction_data2[:, :, 44:])
                prediction2 = torch.mean(prediction2, dim = 1)

                chance_win = round((prediction1[0][0].item() + prediction2[0][2].item()) / 2, 3)
                chance_draw = round((prediction1[0][1].item() + prediction2[0][1].item()) / 2, 3)
                chance_loss = round((prediction1[0][2].item() + prediction2[0][0].item()) / 2, 3)

                t_chance_win += chance_win
                t_chance_draw += chance_draw
                t_chance_loss += chance_loss

                chance_win = t_chance_win
                chance_draw = t_chance_draw
                chance_loss = t_chance_loss

                # chance_win = round(prediction1[0][0].item(), 2)
                # chance_draw = round(prediction1[0][1].item(), 2)
                # chance_loss = round(prediction1[0][2].item(), 2)

                #print(chance_win, chance_draw, chance_loss)

                est_return1 = round(chance_win * float(odds1), 3)
                est_return2 = round(chance_draw * float(odds2), 3)
                est_return3 = round(chance_loss * float(odds3), 3)

                min_return = 1
                max_return = 10

                if est_return1 > min_return and est_return1 > est_return2 and est_return1 > est_return3 and est_return1 < max_return: #est_return1 > min_return and 
                    chosen_bet = 0
                    chosen_odds = odds1
                    chosen_est = est_return1
                elif est_return2 > min_return and est_return2 >= est_return1 and est_return2 > est_return3 and est_return2 < max_return:
                    chosen_bet = 1
                    chosen_odds = odds2
                    chosen_est = est_return2
                elif est_return3 > min_return and est_return3 >= est_return1 and est_return3 >= est_return2 and est_return3 < max_return:
                    chosen_bet = 2
                    chosen_odds = odds3
                    chosen_est = est_return3
                else:
                    chosen_bet = -1
                    chosen_odds = -1
                    chosen_est = 0

                if chosen_bet == -1:
                    pass
                elif chosen_bet == int(result):
                    i += 1
                    returns += chosen_odds
                    t_est += chosen_est
                    num_correct += 1
                else:
                    i += 1
                    t_est += chosen_est

                # print(team1, team2)
                # print(chosen_bet, result)
                # print(chance_win, chance_draw, chance_loss)
                # print(est_return1, est_return2, est_return3)
                # print()
                #print(uncertainty)

            except Exception as e:
                #print(e)
                pass

        return round(t_est / i, 3)

    def train(self, training_data):
        num_epochs = 15
        lr = 3e-2 #5e-7 #1e-7 #3e-6 #8e-7 # Learning rate
        wd = 0 #1e-6 #3e-6 # Weight decay
        batch_size = 500
        warmup_steps = 0
        seq_len = 12
        n_features = 44
        n_out = 14
        lowest_est = 2

        start_time = time.time()
        plot_loss = np.empty((num_epochs), dtype = float)
        plot_est_returns = np.empty((num_epochs), dtype = float)
        plot_scaler = np.empty((num_epochs), dtype = float)

        X = torch.load('training_data.pt').to(self.device)
        y = torch.load('training_targets.pt').to(self.device)

        # print(X)
        # print(y)

        # X shape: (18911, 12, 88)
        # y shape: (18911, 3)

        self.means = torch.mean(X, dim = 0)
        self.stds = torch.std(X, dim = 0)
        
        X -= self.means
        X /= self.stds

        train_data = []
        for i in range(len(X)):
            train_data.append([X[i], y[i]])

        train_set, val_set = torch.utils.data.random_split(train_data, [17000, X.shape[0] - 17000]) # Splits the training data into a train set and a validation set

        train_dataloader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = 0)
        val_dataloader = torch.utils.data.DataLoader(val_set, batch_size = batch_size, shuffle = False, num_workers = 0)

        base_params = self.model.model.parameters()
        softmax_params = self.model.model.softmax.parameters()

        criterion = nn.MSELoss() #nn.L1Loss() #nn.MSELoss() #nn.CrossEntropyLoss()
        # optimizer = optim.Adam([
        #         {'params': base_params},
        #         {'params': softmax_params, 'lr': lr * 2e+4}
        #     ], lr = lr, weight_decay = wd) # 5e-6   1e-8 #1e-3

        optimizer = optim.Adam(base_params, lr = lr, weight_decay = wd) # 5e-6   1e-8 #1e-3

        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [], gamma = 1e-1) #3, 6, 10, 20, 30, 40, 50

        for epoch in range(num_epochs):
            # Learning rate warmup
            if epoch < warmup_steps:
                for g in optimizer.param_groups:
                    g['lr'] = lr * ((epoch + 1) / warmup_steps)

            epoch_start = time.time()

            print('Epoch: ', epoch)
            print(self.model.model.softmax.scaler.item())
            train_loss = 0.0
            self.model.train()

            for batch_idx, (data, labels) in enumerate(train_dataloader):               
                data = data.float().to(self.device)
                labels = labels.to(self.device)
                labels = einops.repeat(labels, 'b n -> b s n', s = seq_len)

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
                    labels = einops.repeat(labels, 'b n -> b s n', s = seq_len)

                    team1 = data[:, :, :44] # shape: (500, 12, 44)
                    team2 = data[:, :, 44:]
                    
                    target = self.model(team1, team2)
                    #print(target[0])
                    loss = criterion(target, labels).float()
                    valid_loss = loss.item() * data.size(0)

            scheduler.step()

            # valid_accuracy = check_accuracy(val_dataloader)
            # print(valid_accuracy, '% Validation Accuracy')
            print('Validation Loss: ', valid_loss)
            est_returns = self.test()
            plot_est_returns[epoch] = est_returns #valid_loss
            plot_loss[epoch] = valid_loss #valid_loss
            plot_scaler[epoch] = self.model.model.softmax.scaler.item()
            print('Epoch time: %s seconds' % round(time.time() - epoch_start, 2))

            if est_returns < lowest_est:
                lowest_est = est_returns
                torch.save(self.model, 'posttrained_model.pt')

        print('Finished in %s seconds' % round(time.time() - start_time, 1))
        plt.plot(plot_loss)
        plt.ylabel('Validation Loss') #('Validation Loss')
        plt.xlabel('Epoch')
        plt.show()

        plt.plot(plot_scaler)
        plt.ylabel('Softmax Scaler') #('Validation Loss')
        plt.xlabel('Epoch')
        plt.show()

        plt.plot(plot_est_returns)
        plt.ylabel('Esitmated Returns') #('Validation Loss')
        plt.xlabel('Epoch')
        plt.show()

        #torch.save(self.model.state_dict(), 'trained_model.pickle')
        torch.save(self.model, 'model.pt')
        print('Saved model to .pickle file')
        torch.save(self.means, 'trained_means.pt')
        torch.save(self.stds, 'trained_stds.pt')
        print('Saved means and standard deviations')
        print('Lowest est_returns:', lowest_est)

if __name__ == '__main__':
    data = Data()
    model = PostTrainingModel()
    means = torch.zeros((40))
    stds = torch.zeros((40))
    pretrain_data = data.pretrain_data
    p = Train(model, means, stds)
    p.train(pretrain_data)