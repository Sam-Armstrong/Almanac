import torch
import torch.nn as nn
import numpy as np
from Predictor import Predictor
from Data import Data
import datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_model():
    data = Data()
    model = torch.load('BSF_best_model.pt')
    model.eval()
    model.model.eval()
    model.model.encoder_model.eval()

    means = torch.load('trained_means.pt').to(device)
    stds = torch.load('trained_stds.pt').to(device)
    means = means.reshape(1, means.shape[0], means.shape[1])
    stds = stds.reshape(1, stds.shape[0], stds.shape[1])
    team1 = ''
    team2 = ''
    date = datetime.datetime.today().strftime('%Y-%m-%d')

    while team1 != 'exit' and team2 != 'exit':
        team1 = str(input('Home Team: '))
        team2 = str(input('Away Team: '))

        prediction_data1 = data.findTimeSeries(team1, team2, date)
        prediction_data2 = data.findTimeSeries(team2, team1, date)

        # if index == 1:
        #     print(prediction_data1)

        model.eval()
        model.evaluate()
        prediction_data1 = prediction_data1.to(device)
        prediction_data1 -= means
        prediction_data1 /= stds
        prediction1 = model.forward(prediction_data1[:, :, :44], prediction_data1[:, :, 44:])
        #prediction1 = torch.mean(prediction1, dim = 1)
        prediction1 = prediction1[:, -1, :]

        prediction_data2 = prediction_data2.to(device)
        prediction_data2 -= means
        prediction_data2 /= stds
        prediction2 = model.forward(prediction_data2[:, :, :44], prediction_data2[:, :, 44:])
        #prediction2 = torch.mean(prediction2, dim = 1)
        prediction2 = prediction2[:, -1, :]

        chance_win = round((prediction1[0][0].item() + prediction2[0][2].item()) / 2, 3)
        chance_draw = round((prediction1[0][1].item() + prediction2[0][1].item()) / 2, 3)
        chance_loss = round((prediction1[0][2].item() + prediction2[0][0].item()) / 2, 3)

        print(chance_win, chance_draw, chance_loss)

if __name__ == '__main__':
    run_model()