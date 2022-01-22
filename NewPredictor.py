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
import pandas
from PretrainingModel import Model
from PreTraining import PreTrain
from FinetuneModel import FinetunedModel

# Train using all features of the next match as the labels?

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def MAPELoss(output, target):
    loss = torch.mean(torch.abs((target - output) / target))
    return loss

def test(predictor, test_odds = 'TestOdds1.csv'):        
    data = Data()

    df = pandas.read_csv(test_odds)
    i = 0
    returns = 0
    t_est = 0
    num_correct = 0
    num_matches = 0

    for index, row in df.iterrows():
        try:
            date = row[0]
            team1 = row[1]
            team2 = row[2]
            result = row[3]
            odds1 = row[4]
            odds2 = row[5]
            odds3 = row[6]

            prediction_data1 = [0] + (data.findTeamStats(team1, date)) + (data.findTeamStats(team2, date))
            prediction_data2 = [1] + (data.findTeamStats(team2, date)) + (data.findTeamStats(team1, date))
            prediction_data1 = np.array([prediction_data1])
            prediction_data2 = np.array([prediction_data2])
            prediction1 = predictor.predict(prediction_data1)
            prediction2 = predictor.predict(prediction_data2)

            chance_win = round((prediction1[0][0].item() + prediction2[0][2].item()) / 2, 3)
            chance_draw = round((prediction1[0][1].item() + prediction2[0][1].item()) / 2, 3)
            chance_loss = round((prediction1[0][2].item() + prediction2[0][0].item()) / 2, 3)

            est_return1 = round(chance_win * float(odds1), 3)
            est_return2 = round(chance_draw * float(odds2), 3)
            est_return3 = round(chance_loss * float(odds3), 3)

            min_return = 1

            if est_return1 > min_return and est_return1 > est_return2 and est_return1 > est_return3:
                chosen_bet = 0
                chosen_odds = odds1
                chosen_est = est_return1
            elif est_return2 > min_return and est_return2 >= est_return1 and est_return2 > est_return3:
                chosen_bet = 1
                chosen_odds = odds2
                chosen_est = est_return2
            elif est_return3 > min_return and est_return3 >= est_return1 and est_return3 >= est_return2:
                chosen_bet = 2
                chosen_odds = odds3
                chosen_est = est_return3
            else:
                chosen_bet = -1
                chosen_odds = -1
                chosen_est = 0

            # if chance_win > chance_draw and chance_win > chance_loss:
            #     chosen_bet = 0
            #     chosen_odds = odds1
            # elif chance_draw >= chance_win and chance_draw > chance_loss:
            #     chosen_bet = 1
            #     chosen_odds = odds2
            # elif chance_loss >= chance_draw and chance_loss >= chance_win:
            #     chosen_bet = 2
            #     chosen_odds = odds3
            # else:
            #     chosen_bet = -1
            #     chosen_odds = -1

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

        except Exception as e:
            pass

    return returns, i

class Predictor:
    def __init__(self, model_name = 'model.pickle', eval = False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.means = torch.zeros((29)).to(self.device)
        self.stds = torch.zeros((29)).to(self.device)
        
        # Pretrained network
        self.model = FinetunedModel().to(self.device)
        
        try:
            #self.model.load_state_dict(torch.load(model_name))
            self.means = torch.load('means.pt')
            self.stds = torch.load('stds.pt')
        except:
            print('No previously trained model found')
            pass

        if eval == True:
            self.model.eval()

    # Method for making predictions using the model
    def predict(self, prediction_data): # prediction_data2 ??
        self.model.eval()
        prediction_data = torch.from_numpy(prediction_data).float().to(self.device)
        prediction_data -= self.means.to(device)
        prediction_data /= self.stds.to(device)
        # print(prediction_data)
        # print(prediction_data.shape)
        prediction = self.model(prediction_data)
        #print(prediction)
        return prediction

    def pretrain(self, training_data):
        p = PreTrain(self.model, self.means, self.stds)
        p.train(training_data)

    # Method for training the model using a given set of data
    def train(self, training_data):
        self.model.train()

        best_returns = 0
        print(len(training_data))

        #self.model = Model().to(self.device)

        num_epochs = 5
        lr = 5e-6 #5e-6 #5e-7 #1e-7 #3e-6 #8e-7 # Learning rate
        wd = 0 #1e-6 #3e-6 # Weight decay
        batch_size = 500
        warmup_steps = 10

        start_time = time.time()
        plot_data = np.empty((num_epochs), dtype = float)
        returns_data = np.empty((num_epochs, 2), dtype = float)

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

        criterion = nn.MSELoss() #nn.KLDivLoss() #nn.L1Loss() #nn.MSELoss() #nn.CrossEntropyLoss()
        optimizer = optim.Adam(params, lr = lr, weight_decay = wd) # 5e-6   1e-8 #1e-3
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [], gamma = 1e-1) #3, 6, 10, 20, 30, 40, 50

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
                    print(target[0])
                    loss = criterion(target, labels).float()
                    valid_loss = loss.item() * data.size(0)

            scheduler.step()

            # valid_accuracy = check_accuracy(val_dataloader)
            # print(valid_accuracy, '% Validation Accuracy')
            print('Validation Loss: ', valid_loss)

            returns, num_samples = test(self, 'TestOdds2.csv')
            returns = returns / num_samples
            print('TestOdds2 returns: ', returns)
            print('Num Predictions: ', num_samples)
            returns_data[epoch, 0] = returns

            if returns > best_returns and returns > 1:
                torch.save(self.model.state_dict(), 'best_model.pickle')
                best_returns = returns

            #if returns > 1:
            returns, num_samples = test(self, 'TestOdds1.csv')
            returns = returns / num_samples
            returns_data[epoch, 1] = returns
            print('TestOdds1 returns: ', returns)
            print('Num Predictions: ', num_samples)

            plot_data[epoch] = valid_loss
            print('Epoch time: %s seconds' % round(time.time() - epoch_start, 2))
            print()

        print('Finished in %s seconds' % round(time.time() - start_time, 1))
        plt.plot(plot_data)
        plt.ylabel('Validation Loss')
        plt.xlabel('Epoch')
        plt.show()

        plt.plot(returns_data)
        plt.ylabel('TestOdds Accuracy')
        plt.xlabel('Epoch')
        plt.show()

        torch.save(self.model.state_dict(), 'model.pickle')
        print('Saved model to .pickle file')
        torch.save(self.means, 'means.pt')
        torch.save(self.stds, 'stds.pt')
        print('Saved means and standard deviations')
        print('Best returns: %s' % best_returns)


if __name__ == '__main__':
    predictor = Predictor()
    data = Data()
    training_data = data.training_data
    predictor.train(training_data)