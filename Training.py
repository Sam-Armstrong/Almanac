import torch
import torch.nn as nn
import math
from Model import Model
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

class Train():
    
    def __init__(self, model, means, stds):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.means = means.to(self.device)
        self.max = torch.scalar_tensor(1)
        self.min = torch.scalar_tensor(1)
        self.stds = stds.to(self.device)
        self.data = Data()

        total_params = sum(p.numel() for p in model.parameters())
        print('Number of Learnable Params:', total_params)

    def test(self, prediction_data, odds_result):
        self.model.eval()

        i = 0
        returns = 0
        t_est = 0
        num_correct = 0

        prediction_data -= self.means
        prediction_data /= self.stds

        self.model.eval()
        predictions = self.model(prediction_data[:, :, :44], prediction_data[:, :, 44:])

        for n in range(predictions.shape[0]):
            try:
                prediction = predictions[n] / 100
                chance_win = round(prediction[0].item(), 3)
                chance_draw = round(prediction[1].item(), 3)
                chance_loss = round(prediction[2].item(), 3)
                
                odds1 = odds_result[n, 1].item()
                odds2 = odds_result[n, 2].item()
                odds3 = odds_result[n, 3].item()
                odds_list = [odds1, odds2, odds3]

                est_return1 = round(chance_win * odds1, 3)
                est_return2 = round(chance_draw * odds2, 3)
                est_return3 = round(chance_loss * odds3, 3)

                result = odds_result[n, 0].item()

                min_return = 1
                max_return = 100

                bets = torch.zeros(3)

                if est_return1 > min_return and est_return1 < max_return:
                    bets[0] = est_return1
                if est_return2 > min_return and est_return2 < max_return:
                    bets[1] = est_return2
                if est_return3 > min_return and est_return3 < max_return:
                    bets[2] = est_return3

                chosen_est = torch.max(bets).item()
                if chosen_est < min_return:
                    raise Exception('No value in betting')

                chosen_bet = torch.argmax(bets).item()
                chosen_odds = odds_list[chosen_bet]

                if math.isnan(result) == False:
                    if chosen_est < min_return:
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
                #print(e)
                pass

        self.model.train()
        self.model.train()
        self.model.encoder_model.train()
        for block in self.model.blocks:
            block.train()
        
        return round(t_est / i, 3), round(returns / i, 3), 100 * num_correct / i

    def train(self): 
        num_epochs = 200
        lr = 3e-6 # Learning rate
        wd = lr * 10 # Weight decay
        batch_size = 1000
        warmup_steps = 5
        seq_len = 12
        n_features = 44
        n_out = 14
        lowest_est = 2
        highest_act = 1

        start_time = time.time()
        plot_loss = np.empty((num_epochs), dtype = float)
        plot_est_returns = np.empty((num_epochs, 3), dtype = float)
        plot_acc = np.empty((num_epochs), dtype = float)

        X = torch.load('training_data.pt').to(self.device)
        y = torch.load('training_targets.pt').to(self.device)

        X = X.reshape(X.shape[0] * X.shape[1], X.shape[2])
        self.means = torch.mean(X, dim = 0).reshape(88)
        self.stds = torch.std(X, dim = 0).reshape(88)
        self.min = torch.min(X)
        self.max = torch.max(X)
        X = X.reshape(X.shape[0] // 12, 12, 88)
        
        X -= self.means
        X /= self.stds
        
        prediction_data_test = torch.load('test_input_tensor.pt').to(device)
        odds_result_test = torch.load('test_target_tensor.pt').to(device)
        prediction_data_val = torch.load('val_input_tensor.pt').to(device)
        odds_result_val = torch.load('val_target_tensor.pt').to(device)

        train_data = []
        for i in range(len(X)):
            train_data.append([X[i], y[i]])

        train_set, val_set = torch.utils.data.random_split(train_data, [15000, X.shape[0] - 15000]) # Splits the training data into a train set and a validation set

        train_dataloader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = 0)
        val_dataloader = torch.utils.data.DataLoader(val_set, batch_size = batch_size, shuffle = False, num_workers = 0)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr = lr, weight_decay = wd)


        pbar = tqdm(range(num_epochs))
        for epoch in pbar:
            
            # Learning rate warmup
            if epoch < warmup_steps:
                for g in optimizer.param_groups:
                    g['lr'] = lr * ((epoch + 1) / warmup_steps)

            epoch_start = time.time()
            
            train_loss = 0.0
            self.model.train()

            for batch_idx, (data, labels) in enumerate(train_dataloader):               
                data = data.float().to(self.device)
                labels = labels.to(self.device) * 100

                team1 = data[:, :, :44] # shape: (500, 12, 44)
                team2 = data[:, :, 44:]
                scores = self.model(team1, team2) # Runs a forward pass of the model for all the data
                loss = criterion(scores, labels).float() # Calculates the loss of the forward pass using the loss function
                train_loss += loss

                optimizer.zero_grad() # Resets the optimizer gradients to zero for each batch
                loss.backward() # Backpropagates the network using the loss to calculate the local gradients

                optimizer.step() # Updates the network weights and biases

            valid_loss = 0.0
            self.model.eval()

            est_returns1, test_returns, test_accuracy = self.test(prediction_data_test.clone(), odds_result_test.clone())
            est_returns2, val_returns, test_accuracy = self.test(prediction_data_val.clone(), odds_result_val.clone())
            plot_est_returns[epoch, 0] = (est_returns1 + est_returns2) / 2
            plot_est_returns[epoch, 1] = test_returns
            plot_est_returns[epoch, 2] = val_returns

            if est_returns2 < lowest_est:
                lowest_est = est_returns2
                torch.save(self.model, 'best_model.pt')

            if val_returns > highest_act:
                highest_act = val_returns
                torch.save(self.model, 'best_performing_model.pt')

            num_correct = 0
            num_samples = 0

            for batch_idx, (data, labels) in enumerate(val_dataloader):
                with torch.no_grad():
                    data = data.float().to(self.device)
                    labels = labels.to(self.device) * 100

                    team1 = data[:, :, :44] # shape: (500, 12, 44)
                    team2 = data[:, :, 44:]
                    
                    target = self.model(team1, team2)
                    loss = criterion(target, labels).float()
                    valid_loss += loss.item() * data.size(0)

                    num_correct += torch.sum(target * labels).item()
                    num_samples += target.size(0)
            
            valid_acc = num_correct / (num_samples * 100)
            plot_acc[epoch] = valid_acc
            plot_loss[epoch] = valid_loss
            pbar.set_description(f"epoch {epoch + 1}; validation loss {round(valid_loss, 2):.5f}; accuracy average {round(valid_acc, 2)}; lr {lr:e}; epoch_time {round(time.time() - epoch_start, 2):.5f}")



        print('Lowest est_returns:', lowest_est)
        print('Highest act_returns:', highest_act)
        
        # Plots the data
        plt.plot(plot_loss)
        plt.ylabel('Validation Loss')
        plt.xlabel('Epoch')
        plt.show()

        plt.plot(plot_acc)
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.show()

        plt.plot(plot_est_returns)
        plt.ylabel('Est Returns // Actual  Returns')
        plt.xlabel('Epoch')
        plt.show()

        torch.save(self.model, 'model.pt')
        

if __name__ == '__main__':
    data = Data()
    model = Model(150, 12, 5, 250)
    means = torch.zeros((40))
    stds = torch.zeros((40))
    pretrain_data = data.pretrain_data
    p = Train(model, means, stds)
    p.train()