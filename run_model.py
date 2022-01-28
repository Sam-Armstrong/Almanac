import torch
import torch.nn as nn
import numpy as np
from Predictor import Predictor
from Data import Data
import datetime

def run_model():
    data = Data()
    predictor = Predictor(model_name = 'BSF1.pickle', eval = True)
    team1 = ''
    team2 = ''
    date = datetime.datetime.today().strftime('%Y-%m-%d')

    while team1 != 'exit' and team2 != 'exit':
        team1 = str(input('Home Team: '))
        team2 = str(input('Away Team: '))

        prediction_data1 = [0] + (data.findTeamStats(team1, date)) + (data.findTeamStats(team2, date))
        prediction_data2 = [1] + (data.findTeamStats(team2, date)) + (data.findTeamStats(team1, date))

        prediction_data1 = np.array([prediction_data1])
        prediction_data2 = np.array([prediction_data2])

        prediction1 = predictor.predict(prediction_data1)
        prediction2 = predictor.predict(prediction_data2)

        chance_win = round((prediction1[0][0].item() + prediction2[0][2].item()) / 2, 3)
        chance_draw = round((prediction1[0][1].item() + prediction2[0][1].item()) / 2, 3)
        chance_loss = round((prediction1[0][2].item() + prediction2[0][0].item()) / 2, 3)

        print(chance_win * 100, chance_draw * 100, chance_loss * 100)

if __name__ == '__main__':
    run_model()