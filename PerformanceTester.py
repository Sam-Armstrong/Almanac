#from Predictor import Predictor
from Data import Data
import pandas
import numpy as np
import torch
import torch.nn as nn
from adaSoft import adaSoft

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test():
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

    ### Test normalizing to minus the min? Also check how layernorm works (just 0 mean unit variance?)

    # model = Model().to(device)
    # model.load_state_dict(torch.load('trained_model.pickle'))
    # model.eval()
    #model = Model
    model = torch.load('best_performing_model.pt') #torch.load('BSFCurrent2.pt')
    #model.softmax = adaSoft(weight_init = 1) #nn.Softmax(dim = -1)
    model.eval()
    model.model.eval()
    model.model.encoder_model.eval()

    means = torch.load('trained_means.pt').to(device)
    stds = torch.load('trained_stds.pt').to(device)
    means = means.reshape(1, means.shape[0], means.shape[1])
    stds = stds.reshape(1, stds.shape[0], stds.shape[1])
    min = torch.load('min.pt')
    max = torch.load('max.pt')

    guesses = [0, 0, 0]

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

            odds_list = [odds1, odds2, odds3]

            prediction_data1 = data.findTimeSeries(team1, team2, date)
            prediction_data2 = data.findTimeSeries(team2, team1, date)

            # if index == 1:
            #     print(prediction_data1)

            model.eval()
            model.evaluate()
            prediction_data1 = prediction_data1.to(device)
            prediction_data1 -= means
            prediction_data1 /= stds
            #prediction_data1 -= min
            #prediction_data1 /= max
            prediction1 = model.forward(prediction_data1[:, :, :44], prediction_data1[:, :, 44:])
            #prediction1 = torch.mean(prediction1, dim = 1)
            #prediction1 = prediction1[:, -1, :]

            prediction_data2 = prediction_data2.to(device)
            prediction_data2 -= means
            prediction_data2 /= stds
            #prediction_data2 -= min
            prediction2 = model.forward(prediction_data2[:, :, :44], prediction_data2[:, :, 44:])
            #prediction2 = torch.mean(prediction2, dim = 1)
            #prediction2 = prediction2[:, -1, :]

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
            # print(team1, team2)
            print(chance_win, chance_draw, chance_loss)

            est_return1 = round(chance_win * float(odds1), 3)
            est_return2 = round(chance_draw * float(odds2), 3)
            est_return3 = round(chance_loss * float(odds3), 3)

            min_return = 1
            max_return = 100

            bets = torch.zeros(3)

            if est_return1 > min_return and est_return1 < max_return: #est_return1 > min_return and 
                bets[0] = est_return1
            elif est_return2 > min_return and est_return2 < max_return:
                bets[1] = est_return2
            elif est_return3 > min_return and est_return3 < max_return:
                bets[2] = est_return3
            else:
                chosen_bet = -1
                chosen_odds = -1
                chosen_est = 0

            chosen_est = torch.max(bets).item()
            if chosen_est < min_return:
                raise

            chosen_bet = torch.argmax(bets).item()
            chosen_odds = odds_list[chosen_bet]

            # if est_return1 > min_return and est_return1 > est_return2 and est_return1 > est_return3 and est_return1 < max_return: #est_return1 > min_return and 
            #     chosen_bet = 0
            #     chosen_odds = odds1
            #     chosen_est = est_return1
            # elif est_return2 > min_return and est_return2 >= est_return1 and est_return2 > est_return3 and est_return2 < max_return:
            #     chosen_bet = 1
            #     chosen_odds = odds2
            #     chosen_est = est_return2
            # elif est_return3 > min_return and est_return3 >= est_return1 and est_return3 >= est_return2 and est_return3 < max_return:
            #     chosen_bet = 2
            #     chosen_odds = odds3
            #     chosen_est = est_return3
            # else:
            #     chosen_bet = -1
            #     chosen_odds = -1
            #     chosen_est = 0

            # if chance_win > chance_draw and chance_win > chance_loss:
            #     chosen_bet = 0
            #     chosen_odds = odds1
            #     chosen_est = est_return1
            # elif chance_draw >= chance_win and chance_draw > chance_loss:
            #     chosen_bet = 1
            #     chosen_odds = odds2
            #     chosen_est = est_return2
            # elif chance_loss >= chance_draw and chance_loss >= chance_win:
            #     chosen_bet = 2
            #     chosen_odds = odds3
            #     chosen_est = est_return3
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

            guesses[chosen_bet] += 1

            # print(team1, team2)
            # print(chosen_bet, result)
            # print(chance_win, chance_draw, chance_loss)
            # print(est_return1, est_return2, est_return3)
            # print()
            #print(uncertainty)

        except Exception as e:
            #print(e)
            pass

    print('i: ', i)
    print('Number Correct: ', num_correct)
    print('Accuracy: %s' % round(num_correct * 100 / i, 2))
    print('Returns: ', round(returns, 3))
    print('Expected Return: ', round(t_est / i, 3))
    print('Average Return: ', round(returns / i, 3))
    print('Guesses:', guesses)




if __name__ == '__main__':
    test()