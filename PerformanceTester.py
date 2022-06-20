import math
from Data import Data
import pandas
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test():
    data = Data()

    offset = 1
    odds_df = pandas.read_csv('test_odds.csv')
    i = 0
    returns = 0
    t_est = 0
    num_correct = 0
    num_matches = 0
    
    model = torch.load('model.pt')
    model.no_transformer_grad()
    model.eval()
    model.model.eval()
    model.model.encoder_model.eval()
    for block in model.model.blocks:
        block.eval()

    means = torch.load('trained_means.pt').to(device)
    stds = torch.load('trained_stds.pt').to(device)
    means = means.repeat(1, 12, 1)
    stds = stds.repeat(1, 12, 1)

    guesses = [0, 0, 0]

    for n, (index, row) in enumerate(odds_df.iterrows()):
        try:
            date = row[0 + offset]
            team1 = row[1 + offset]
            team2 = row[2 + offset]
            result = row[3 + offset]
            odds1 = row[4 + offset]
            odds2 = row[5 + offset]
            odds3 = row[6 + offset]

            odds_list = [odds1, odds2, odds3]
            string_result_array = [team1, 'Draw', team2]

            prediction_data1 = data.findTimeSeries(team1, team2, date).to(device)

            prediction_data1 -= means
            prediction_data1 /= stds
            prediction1 = model(prediction_data1[:, :, :44], prediction_data1[:, :, 44:]) / 100

            chance_win = round(prediction1[0, 0].item(), 3)
            chance_draw = round(prediction1[0, 1].item(), 3)
            chance_loss = round(prediction1[0, 2].item(), 3)

            print(chance_win, chance_draw, chance_loss)

            est_return1 = round(chance_win * odds1, 3)
            est_return2 = round(chance_draw * odds2, 3)
            est_return3 = round(chance_loss * odds3, 3)

            min_return = 1 #1
            max_return = 100 #1.5

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

            else:
                print(team1, 'vs', team2, '¦ Bet on:', string_result_array[chosen_bet])

            guesses[chosen_bet] += 1


        except Exception as e:
            print(team1, 'vs', team2, 'failed due to:', e)
            pass

    try:
        print('i: ', i)
        print('Number Correct: ', num_correct)
        print('Accuracy: %s' % round(num_correct * 100 / i, 2))
        print('Returns: ', round(returns, 3))
        print('Expected Return: ', round(t_est / i, 3))
        print('Average Return: ', round(returns / i, 3))
        print('Guesses:', guesses)
    except:
        print('Nothing else to see')

if __name__ == '__main__':
    test()