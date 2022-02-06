"""
Author: Sam Armstrong
Date: 2021
Description: Predictor model using self-attention layers
"""

import torch
from Model import Model

# Train using all features of the next match as the labels?

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Predictor:
    def __init__(self, model_name = 'trained_model.pickle', eval = True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = Model().to(self.device)
        self.model.load_state_dict(torch.load('trained_model.pickle'))
        self.model.eval()
        
        try:
            #self.model.load_state_dict(torch.load(model_name))
            self.means = torch.load('trained_means.pt').to(self.device)
            self.stds = torch.load('trained_stds.pt').to(self.device)
            self.means = self.means.reshape(1, self.means.shape[0], self.means.shape[1])
            self.stds = self.stds.reshape(1, self.stds.shape[0], self.stds.shape[1])
        except:
            print('No previously trained model found')
            pass
            

    # Method for making predictions using the model
    def predict(self, prediction_data): # prediction_data2 ??
        self.model.eval()
        self.model.evaluate()
        prediction_data = prediction_data.to(device)
        prediction_data -= self.means
        prediction_data /= self.stds
        prediction = self.model(prediction_data[:, :, :44], prediction_data[:, :, 44:])
        return prediction
