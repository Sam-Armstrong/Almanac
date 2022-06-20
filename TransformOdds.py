from cgi import test
import pandas


def run_script():
    """
    Script for creating a validation set and test set from all the scraped odds
    """
    odds_dataframe = pandas.read_csv('all_odds.csv')
    validation_dataframe = pandas.DataFrame(columns = ['Date', 'Team 1', 'Team 2', 'Result', 'Odds 1', 'Odds 2', 'Odds 3'])
    test_dataframe = pandas.DataFrame(columns = ['Date', 'Team 1', 'Team 2', 'Result', 'Odds 1', 'Odds 2', 'Odds 3'])

    for index, row in odds_dataframe.iterrows():
        if index % 2 == 0:
            validation_dataframe.loc[len(validation_dataframe)] = row
        else:
            test_dataframe.loc[len(validation_dataframe)] = row

    
    validation_dataframe.to_csv('validation_odds.csv')
    test_dataframe.to_csv('test_odds.csv')



if __name__ == '__main__':
    run_script()
    print('Complete. ')