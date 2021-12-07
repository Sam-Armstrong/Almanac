from Predictor import Predictor
from Data import Data
import pandas
import numpy as np

def test():
    predictor = Predictor()
    data = Data()

    df = pandas.read_csv('TestOdds2.csv')
    i = 0
    returns = 0
    t_est = 0

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

            # uncertainty = abs(prediction1[0][0].item() - prediction2[0][2].item()) + abs(prediction1[0][1].item() + prediction2[0][1].item()) + abs(prediction1[0][2].item() + prediction2[0][0].item())

            # Doesn't seem to help
            # if uncertainty < 1:
            #     raise

            est_return1 = round(chance_win * float(odds1), 3)
            est_return2 = round(chance_draw * float(odds2), 3)
            est_return3 = round(chance_loss * float(odds3), 3)

            if est_return1 > 1 and est_return1 > est_return2 and est_return1 > est_return3:
                chosen_bet = 0
                chosen_odds = odds1
                chosen_est = est_return1
            elif est_return2 > 1 and est_return2 >= est_return1 and est_return2 > est_return3:
                chosen_bet = 1
                chosen_odds = odds2
                chosen_est = est_return2
            elif est_return3 > 1 and est_return3 >= est_return1 and est_return3 >= est_return2:
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
            pass

    print('i: ', i)
    print('Returns: ', round(returns, 3))
    print('Expected Return: ', round(t_est / i, 3))
    print('Average Return: ', round(returns / i, 3))




if __name__ == '__main__':
    test()