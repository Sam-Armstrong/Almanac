"""
Author: Sam Armstrong
Date: 2021
"""

import bs4, requests, datetime, pandas, time, torch, einops
import numpy as np

# Calculates the days between a given date and the current date
def calculateDaysSince(date):
    split_date = date.split('-')
    year = split_date[0]
    month = split_date[1]
    day = split_date[2]

    a = datetime.date(int(year), int(month), int(day))
    b = datetime.date.today()
    days_since = b - a
    days_since = str(days_since)

    if len(days_since.split(' ')) > 1:
        days_since = int(days_since.split(' ')[0])
    else:
        days_since = 0

    return days_since

class Data:
    def __init__(self):
        try:
            self.match_results = pandas.read_csv('MatchResults.csv')
        except:
            self.match_results = None
        
        try:
            self.match_stats = pandas.read_csv('MatchStats.csv')
        except:
            self.match_stats = None

        try:
            self.training_data = pandas.read_csv('TrainingData.csv')
        except:
            self.training_data = None

        try:
            self.pretrain_data = pandas.read_csv('PretrainData.csv')
        except:
            self.pretrain_data = None

        try:
            self.time_series = pandas.read_csv('TimeSeriesData.csv')
        except:
            pass


    def findMatchStats(self, team_name, date):
        #n_include = 7 # The number of past matches to calculate the team average from

        days_since_match = calculateDaysSince(date)
        team_data = self.match_stats[self.match_stats['Team 1'].str.contains(team_name)]
        match_data = team_data[team_data['Date'].str.contains(date)]
        stats = []

        for index, row in match_data.iterrows():
            if row[1] == date:
                goals = row[3]
                goals_against = row[4]
                pos = row[5]
                shots_on_t = row[6]
                att_shots = row[7]
                shot_acc = row[8]
                sot_against = row[9]
                att_shots_against = row[10]
                saves = row[11]
                save_acc = row[12]
                fouls = row[13]
                fouls_against = row[14]
                corners = row[15]
                corners_against = row[16]

        stats.append(goals)
        stats.append(goals_against)
        stats.append(pos)
        stats.append(shots_on_t)
        stats.append(att_shots)
        stats.append(shot_acc)
        stats.append(sot_against)
        stats.append(att_shots_against)
        stats.append(saves)
        stats.append(save_acc)
        stats.append(fouls)
        stats.append(fouls_against)
        stats.append(corners)
        stats.append(corners_against)

        return stats

    def findTeamStats(self, team_name, date):
        n_include = 5 # The number of past matches to calculate the team average from

        days_since_match = calculateDaysSince(date)
        team_data = self.match_stats[self.match_stats['Team 1'].str.contains(team_name)]
        average_data = []

        goals = 0
        goals_against = 0
        pos = 0
        shots_on_t = 0
        att_shots = 0
        shot_acc = 0
        sot_against = 0
        att_shots_against = 0
        saves = 0
        save_acc = 0
        fouls = 0
        fouls_against = 0
        corners = 0
        corners_against = 0
        #offsides = 0

        i = 0

        for index, row in team_data.iterrows():
            match_date = row[1]
            #print(match_date) ##

            days_since = calculateDaysSince(match_date)
            # print('days_since:', days_since)
            # print('days_since_match:', days_since_match)

            if days_since > days_since_match and i < n_include:
                goals += row[3]
                goals_against += row[4]
                pos += row[5]
                shots_on_t += row[6]
                att_shots += row[7]
                shot_acc += row[8]
                sot_against += row[9]
                att_shots_against += row[10]
                saves += row[11]
                save_acc += row[12]
                fouls += row[13]
                fouls_against += row[14]
                corners += row[15]
                corners_against += row[16]
                i += 1

        if i != n_include:
            raise Exception('Not enough data available for this team')
        # else:
        #     print(team_name)
        #     print('Enough data!')

        average_data.append(goals)
        average_data.append(goals_against)
        average_data.append(pos)
        average_data.append(shots_on_t)
        average_data.append(att_shots)
        average_data.append(shot_acc)
        average_data.append(sot_against)
        average_data.append(att_shots_against)
        average_data.append(saves)
        average_data.append(save_acc)
        average_data.append(fouls)
        average_data.append(fouls_against)
        average_data.append(corners)
        average_data.append(corners_against)

        return average_data

    def updateData(self):
        date_list = []
        league_list = ['Premier-League', 'La-Liga', 'Bundesliga', 'Serie-A', 'Ligue-1', 'Major-League-Soccer', 'Championship', 'League-One', 'Primeira-Liga']

        results_dataframe = pandas.DataFrame(columns = ['Date', 'Team 1', 'Team 2', 'Result'])
        stats_dataframe = pandas.DataFrame(columns = ['Date', 'Team 1', 'Goals', 'Goals Against', 'Possession', 'Shots on Target', 'Attempted Shots', 'Shot Accuracy', 'SoT Against',
                                                      'Att Shots Against', 'Saves', 'Save Accuracy', 'Fouls', 'Fouls Against', 'Corners', 'Corners Against', 'Offsides']) # 17
        all_results = []
        all_stats = []

        previous_date = self.match_results.iloc[0, 1]

        # Gets the n - 1 previous dates, up to the most recent date in the current data
        for i in range(1, 3000): # (1, n)
            next_date = str(datetime.date.fromordinal(datetime.date.today().toordinal() - i))

            if next_date != previous_date:
                date_list.append(next_date)
            else:
                break # Once all dates have been found up to the last date in the current data the loop breaks

        print('Scraping Data...')
        
        url_list = []
        dates = []

        for date in date_list:
            print('https://fbref.com/en/matches/' + date)
            page = requests.get('https://fbref.com/en/matches/' + date)
            soup = bs4.BeautifulSoup(page.content, 'lxml')

            for td in soup.findAll('td', attrs = {'data-stat': 'match_report'}):
                for a in td.findAll('a', href = True):
                    new_url = a['href']
                    in_league = False
                    for league in league_list:
                        if league in new_url:
                            in_league = True

                    if new_url not in url_list and in_league == True:
                        url_list.append('https://fbref.com/' + new_url)
                        dates.append(date)

        url_list_len = len(url_list)
        x = 0
        matches = [] #Remove

        for i, url in enumerate(url_list):
            print(i + 1, '/', url_list_len)
            #print(url)
            date = dates[i]

            # Team 1, Team 2, Score 1, Score 2, 
            team1_stats = []
            team2_stats = []
            match_result = []
            

            try:
                page = requests.get(url)
                soup = bs4.BeautifulSoup(page.content, 'lxml')

                match_result.append(date)

                teams = []
                for a in soup.findAll('a', href = True, attrs = {'itemprop' : 'name'}):
                    match_result.append(a.text)
                    teams.append(a.text)

                scores = []

                for div in soup.findAll('div', attrs = {'class': 'score'}):
                    scores.append(int(div.text))

                if scores[0] > scores[1]:
                    result = 0 # Win
                elif scores[0] < scores[1]:
                    result = 2 # Loss
                else:
                    result = 1 # Draw

                match_result.append(result)

                numbers = []
                for div in soup.findAll('div', attrs = {'id': 'team_stats'}):
                    for td in div.findAll('td'):
                        text = td.text.replace('\n', '')
                        text = text.replace('%', '')
                        
                        for word in text.split():
                            if word.isdigit():
                                numbers.append(int(word))

                if len(numbers) == 20:
                    pos1 = numbers[0]
                    pos2 = numbers[1]

                    shots_on_t1 = numbers[8]
                    att_shots1 = numbers[9]
                    shot_acc1 = numbers[10]
                    shot_acc2 = numbers[11]
                    shots_on_t2 = numbers[12]
                    att_shots2 = numbers[13]

                    saves1 = numbers[14]
                    save_acc1 = numbers[16]
                    save_acc2 = numbers[17]
                    saves2 = numbers[18]

                elif len(numbers) == 14:
                    pos1 = numbers[0]
                    pos2 = numbers[1]

                    shots_on_t1 = numbers[2]
                    att_shots1 = numbers[3]
                    shot_acc1 = numbers[4]
                    shot_acc2 = numbers[5]
                    shots_on_t2 = numbers[6]
                    att_shots2 = numbers[7]

                    saves1 = numbers[8]
                    save_acc1 = numbers[10]
                    save_acc2 = numbers[11]
                    saves2 = numbers[12]

                else:
                    raise

                
                numbers = []
                for d in soup.findAll('div', attrs = {'id': 'team_stats_extra'}):
                    for div in d.findAll('div'):
                        text = div.text
                        
                        for word in text.split():
                            if word.isdigit():
                                numbers.append(int(word))

                if len(numbers) == 22:
                    fouls1 = numbers[0]
                    fouls2 = numbers[1]
                    corners1 = numbers[2]
                    corners2 = numbers[3]
                    offsides1 = numbers[16]
                    offsides2 = numbers[17]

                elif len(numbers) == 6:
                    fouls1 = numbers[0]
                    fouls2 = numbers[1]
                    corners1 = numbers[2]
                    corners2 = numbers[3]
                    offsides1 = numbers[4]
                    offsides2 = numbers[5]

                team1_stats.append(date)
                team1_stats.append(teams[0])
                team1_stats.append(scores[0])
                team1_stats.append(scores[1])
                team1_stats.append(pos1)
                team1_stats.append(shots_on_t1)
                team1_stats.append(att_shots1)
                team1_stats.append(shot_acc1)
                team1_stats.append(shots_on_t2)
                team1_stats.append(att_shots2) # Shots against them
                team1_stats.append(saves1)
                team1_stats.append(save_acc1)
                team1_stats.append(fouls1)
                team1_stats.append(fouls2)
                team1_stats.append(corners1)
                team1_stats.append(corners2)
                team1_stats.append(offsides1)

                team2_stats.append(date)
                team2_stats.append(teams[1])
                team2_stats.append(scores[1])
                team2_stats.append(scores[0])
                team2_stats.append(pos2)
                team2_stats.append(shots_on_t2)
                team2_stats.append(att_shots2)
                team2_stats.append(shot_acc2)
                team2_stats.append(shots_on_t1)
                team2_stats.append(att_shots1) # Shots against them
                team2_stats.append(saves2)
                team2_stats.append(save_acc2)
                team2_stats.append(fouls2)
                team2_stats.append(fouls1)
                team2_stats.append(corners2)
                team2_stats.append(corners1)
                team2_stats.append(offsides2)

                all_results.append(match_result)
                all_stats.append(team1_stats)
                all_stats.append(team2_stats)
                

            except Exception as e:
                pass

        for data in all_results:
            df_len = len(results_dataframe)
            results_dataframe.loc[df_len] = data
        
        frames = [results_dataframe, self.match_results]
        result = pandas.concat(frames)
        #results_dataframe.append(self.match_results, ignore_index = True) # Adds the new data to the existing data
        result.to_csv('MatchResults.csv') # Saves all the data to a CSV (overwrites existing file)

        for data in all_stats:
            df_len = len(stats_dataframe)
            stats_dataframe.loc[df_len] = data

        frames = [stats_dataframe, self.match_stats]
        result = pandas.concat(frames)
        #stats_dataframe.append(self.match_stats, ignore_index = True)
        result.to_csv('MatchStats.csv')

    def getData(self):
        date_list = []
        league_list = ['Premier-League', 'La-Liga', 'Bundesliga', 'Serie-A', 'Ligue-1', 'Major-League-Soccer', 'Championship', 'League-One', 'Primeira-Liga']

        results_dataframe = pandas.DataFrame(columns = ['Date', 'Team 1', 'Team 2', 'Result'])
        stats_dataframe = pandas.DataFrame(columns = ['Date', 'Team 1', 'Goals', 'Goals Against', 'Possession', 'Shots on Target', 'Attempted Shots', 'Shot Accuracy', 'SoT Against',
                                                      'Att Shots Against', 'Saves', 'Save Accuracy', 'Fouls', 'Fouls Against', 'Corners', 'Corners Against', 'Offsides']) # 17
        all_results = []
        all_stats = []

        # Gets the n - 1 previous dates
        for i in range(1, 3000): # (1, n)
            date_list.append(str(datetime.date.fromordinal(datetime.date.today().toordinal() - i)))

        print('Scraping Data... (This could take a few hours)')
        
        url_list = []
        dates = []

        for date in date_list:
            print('https://fbref.com/en/matches/' + date)
            page = requests.get('https://fbref.com/en/matches/' + date)
            soup = bs4.BeautifulSoup(page.content, 'lxml')

            for td in soup.findAll('td', attrs = {'data-stat': 'match_report'}):
                for a in td.findAll('a', href = True):
                    new_url = a['href']
                    in_league = False
                    for league in league_list:
                        if league in new_url:
                            in_league = True

                    if new_url not in url_list and in_league == True:
                        url_list.append('https://fbref.com/' + new_url)
                        dates.append(date)

        url_list_len = len(url_list)
        x = 0
        matches = [] #Remove

        for i, url in enumerate(url_list):
            print(i + 1, '/', url_list_len)
            #print(url)
            date = dates[i]

            # Team 1, Team 2, Score 1, Score 2, 
            team1_stats = []
            team2_stats = []
            match_result = []
            

            try:
                page = requests.get(url)
                soup = bs4.BeautifulSoup(page.content, 'lxml')

                match_result.append(date)

                teams = []
                for a in soup.findAll('a', href = True, attrs = {'itemprop' : 'name'}):
                    match_result.append(a.text)
                    teams.append(a.text)

                scores = []

                for div in soup.findAll('div', attrs = {'class': 'score'}):
                    scores.append(int(div.text))

                if scores[0] > scores[1]:
                    result = 0 # Win
                elif scores[0] < scores[1]:
                    result = 2 # Loss
                else:
                    result = 1 # Draw

                match_result.append(result)

                numbers = []
                for div in soup.findAll('div', attrs = {'id': 'team_stats'}):
                    for td in div.findAll('td'):
                        text = td.text.replace('\n', '')
                        text = text.replace('%', '')
                        
                        for word in text.split():
                            if word.isdigit():
                                numbers.append(int(word))

                if len(numbers) == 20:
                    pos1 = numbers[0]
                    pos2 = numbers[1]

                    shots_on_t1 = numbers[8]
                    att_shots1 = numbers[9]
                    shot_acc1 = numbers[10]
                    shot_acc2 = numbers[11]
                    shots_on_t2 = numbers[12]
                    att_shots2 = numbers[13]

                    saves1 = numbers[14]
                    save_acc1 = numbers[16]
                    save_acc2 = numbers[17]
                    saves2 = numbers[18]

                elif len(numbers) == 14:
                    pos1 = numbers[0]
                    pos2 = numbers[1]

                    shots_on_t1 = numbers[2]
                    att_shots1 = numbers[3]
                    shot_acc1 = numbers[4]
                    shot_acc2 = numbers[5]
                    shots_on_t2 = numbers[6]
                    att_shots2 = numbers[7]

                    saves1 = numbers[8]
                    save_acc1 = numbers[10]
                    save_acc2 = numbers[11]
                    saves2 = numbers[12]

                else:
                    raise

                
                numbers = []
                for d in soup.findAll('div', attrs = {'id': 'team_stats_extra'}):
                    for div in d.findAll('div'):
                        text = div.text
                        
                        for word in text.split():
                            if word.isdigit():
                                numbers.append(int(word))

                if len(numbers) == 22:
                    fouls1 = numbers[0]
                    fouls2 = numbers[1]
                    corners1 = numbers[2]
                    corners2 = numbers[3]
                    # crosses1 = numbers[4]
                    # crosses2 = numbers[5]
                    # touches1 = numbers[6]
                    # touches2 = numbers[7]
                    # tackles1 = numbers[8]
                    # tackles2 = numbers[9]
                    # interceptions1 = numbers[10]
                    # interceptions2 = numbers[11]
                    # aerial_duels1 = numbers[12]
                    # aerial_duels2 = numbers[13]
                    # clearences1 = numbers[14]
                    # clearences2 = numbers[15]
                    offsides1 = numbers[16]
                    offsides2 = numbers[17]

                elif len(numbers) == 6:
                    fouls1 = numbers[0]
                    fouls2 = numbers[1]
                    corners1 = numbers[2]
                    corners2 = numbers[3]
                    offsides1 = numbers[4]
                    offsides2 = numbers[5]

                team1_stats.append(date)
                team1_stats.append(teams[0])
                team1_stats.append(scores[0])
                team1_stats.append(scores[1])
                team1_stats.append(pos1)
                # team1_stats.append(comp_passes1)
                # team1_stats.append(pass_acc1)
                team1_stats.append(shots_on_t1)
                team1_stats.append(att_shots1)
                team1_stats.append(shot_acc1)
                team1_stats.append(shots_on_t2)
                team1_stats.append(att_shots2) # Shots against them
                # team1_stats.append(comp_passes2)
                # team1_stats.append(pass_acc2)
                team1_stats.append(saves1)
                team1_stats.append(save_acc1)
                team1_stats.append(fouls1)
                team1_stats.append(fouls2)
                team1_stats.append(corners1)
                team1_stats.append(corners2)
                # team1_stats.append(crosses1)
                # team1_stats.append(crosses2)
                # team1_stats.append(touches1)
                # team1_stats.append(touches2)
                # team1_stats.append(tackles1)
                # Not used tackles 2
                # team1_stats.append(interceptions1)
                # team1_stats.append(interceptions2)
                # team1_stats.append(aerial_duels1)
                # team1_stats.append(aerial_duels2)
                # team1_stats.append(clearences1)
                team1_stats.append(offsides1)

                team2_stats.append(date)
                team2_stats.append(teams[1])
                team2_stats.append(scores[1])
                team2_stats.append(scores[0])
                team2_stats.append(pos2)
                team2_stats.append(shots_on_t2)
                team2_stats.append(att_shots2)
                team2_stats.append(shot_acc2)
                team2_stats.append(shots_on_t1)
                team2_stats.append(att_shots1) # Shots against them
                team2_stats.append(saves2)
                team2_stats.append(save_acc2)
                team2_stats.append(fouls2)
                team2_stats.append(fouls1)
                team2_stats.append(corners2)
                team2_stats.append(corners1)
                team2_stats.append(offsides2)

                all_results.append(match_result)
                all_stats.append(team1_stats)
                all_stats.append(team2_stats)
                

            except Exception as e:
                pass

        for data in all_results:
            df_len = len(results_dataframe)
            results_dataframe.loc[df_len] = data

        results_dataframe.to_csv('MatchResults.csv')

        for data in all_stats:
            df_len = len(stats_dataframe)
            stats_dataframe.loc[df_len] = data

        stats_dataframe.to_csv('MatchStats.csv')


    # def createPretrainData(self):
    #     all_data = []

    #     training_data = pandas.DataFrame(columns = ['Home/Away', 'goals1', 'goals_against1', 'pos1', 'shots_on_t1', 'att_shots1', 'shot_acc1', 'sot_against1', 'att_shots_against1', 'saves1', 
    #                                                 'save_acc1', 'fouls1', 'fouls_against1', 'corners1', 'corners_against1', 'goals2', 'goals_against2', 'pos2', 'shots_on_t2', 
    #                                                 'att_shots2', 'shot_acc2', 'sot_against2', 'att_shots_against2', 'saves2', 'save_acc2', 'fouls2', 'fouls_against2', 'corners2', 
    #                                                 'corners_against2', 'tgoals1', 'tgoals_against1', 'tpos1', 'tshots_on_t1', 'tatt_shots1', 'tshot_acc1', 'tsot_against1', 'tatt_shots_against1', 'tsaves1', 
    #                                                 'tsave_acc1', 'tfouls1', 'tfouls_against1', 'tcorners1', 'tcorners_against1', 'tgoals2', 'tgoals_against2', 'tpos2', 'tshots_on_t2', 
    #                                                 'tatt_shots2', 'tshot_acc2', 'tsot_against2', 'tatt_shots_against2', 'tsaves2', 'tsave_acc2', 'tfouls2', 'tfouls_against2', 'tcorners2', 
    #                                                 'tcorners_against2'])

    #     for index, row in self.match_results.iterrows():
    #         print(index, '/', len(self.match_results))
    #         try:
    #             date = row[1]
    #             team1 = row[2].rstrip()
    #             team2 = row[3].rstrip()
    #             result = row[4]

    #             target1 = self.findMatchStats(team1, date)
    #             target2 = self.findMatchStats(team2, date)
    #             team1_useful_data = self.findTeamStats(team1, date)
    #             team2_useful_data = self.findTeamStats(team2, date)

    #             # days_since_match = calculateDaysSince(date)

    #             # if days_since_match < 16:
    #             #     raise

    #             full_list1 = [0] + team1_useful_data + team2_useful_data + target1 + target2 # Size: (14x4) + 1 = 57
    #             full_list2 = [1] + team2_useful_data + team1_useful_data + target2 + target1

    #             all_data.append(full_list1)
    #             all_data.append(full_list2)

    #         except Exception as e:
    #             print(e)
    #             pass # Some matches do not have the correct past data to allow them to be included - hence they are passed over with this except
            
    #     for data in all_data:
    #         df_len = len(training_data)
    #         training_data.loc[df_len] = data

    #     training_data.to_csv('PretrainData.csv')

    def createTrainingData(self):
        all_data = []

        training_data = pandas.DataFrame(columns = ['Home/Away', 'goals1', 'goals_against1', 'pos1', 'shots_on_t1', 'att_shots1', 'shot_acc1', 'sot_against1', 'att_shots_against1', 'saves1', 
                                                    'save_acc1', 'fouls1', 'fouls_against1', 'corners1', 'corners_against1', 'goals2', 'goals_against2', 'pos2', 'shots_on_t2', 
                                                    'att_shots2', 'shot_acc2', 'sot_against2', 'att_shots_against2', 'saves2', 'save_acc2', 'fouls2', 'fouls_against2', 'corners2', 
                                                    'corners_against2', 'Win', 'Draw', 'Loss'])

        for index, row in self.match_results.iterrows():
            print(index, '/', len(self.match_results))
            try:
                date = row[1]
                days_since_match = calculateDaysSince(date)

                if days_since_match < 16:
                    raise

                team1 = row[2].rstrip()
                team2 = row[3].rstrip()
                result = row[4]

                team1_useful_data = self.findTeamStats(team1, date)
                team2_useful_data = self.findTeamStats(team2, date)

                if result == 2:
                    result_array = [0, 0, 1]
                    opposite_array = [1, 0, 0]
                elif result == 1:
                    result_array = [0, 1, 0]
                    opposite_array = [0, 1, 0]
                else:
                    result_array = [1, 0, 0]
                    opposite_array = [0, 0, 1]

                full_list1 = [0] + team1_useful_data + team2_useful_data + result_array # Size: 32
                full_list2 = [1] + team2_useful_data + team1_useful_data + opposite_array

                all_data.append(full_list1)
                all_data.append(full_list2)

            except Exception as e:
                pass # Some matches do not have the correct past data to allow them to be included - hence they are passed over with this except
            
        for data in all_data:
            df_len = len(training_data)
            training_data.loc[df_len] = data

        training_data.to_csv('TrainingData.csv')

    
    ## Need to create function for retriving time series for a certain team at a certain date
    ## Add date?
    ## Filter out dates older than 2016/2017?
    def createTimeSeriesData(self):
        training_data = pandas.DataFrame(columns = ['Date', 'Team 1', 'Team 2', 'Result', 'Home', 'Away', 'goals1', 'goals_against1', 'pos1', 'shots_on_t1', 'att_shots1', 'shot_acc1', 'sot_against1', 'att_shots_against1', 'saves1', 
                                                    'save_acc1', 'fouls1', 'fouls_against1', 'corners1', 'corners_against1', 'goals2', 'goals_against2', 'pos2', 'shots_on_t2', 
                                                    'att_shots2', 'shot_acc2', 'sot_against2', 'att_shots_against2', 'saves2', 'save_acc2', 'fouls2', 'fouls_against2', 'corners2', 
                                                    'corners_against2', 'avg_goals2', 'avg_goals_against2', 'avg_pos2', 'avg_shots_on_t2', 'avg_att_shots2', 'avg_shot_acc2', 
                                                    'avg_sot_against2', 'avg_att_shots_against2', 'avg_saves2', 'avg_save_acc2', 'avg_fouls2', 'avg_fouls_against2', 'avg_corners2', 
                                                    'avg_corners_against2'])

        for index, row in self.match_results.iterrows():
            print(index, '/', len(self.match_results))

            try:
                date = row[1]

                if date < "2016-01-01":
                    raise

                team1 = row[2].rstrip()
                team2 = row[3].rstrip()
                result = row[4]
                teams = [team1, team2]

                for i in range(2):
                    if i == 0:
                        team1 = teams[0]
                        team2 = teams[1]
                    else:
                        team1 = teams[1]
                        team2 = teams[0]

                    team1_stats = self.findMatchStats(team1, date) # 14
                    team2_stats = self.findMatchStats(team2, date) # 14

                    if i == 0:
                        homeaway = [date, team1, team2, result, 1, 0] # 6
                    else:
                        homeaway = [date, team1, team2, result, 0, 1]

                    team2_avg = self.findTeamStats(team2, date) # 14

                    output = homeaway + team1_stats + team2_stats + team2_avg
                    training_data.loc[len(training_data)] = output

            except Exception as e:
                pass

        training_data.to_csv('TimeSeriesData.csv')

    
    def findTimeSeries(self, team, date, max_len): # Returns PyTorch tensor
        try:
            team_data = self.time_series[self.time_series['Team 1'].str.contains(team)]
            seq_tensor = torch.empty((max_len, 44)) # n_features: 44
            
            for index, row in team_data.iterrows():
                match_date = row[1]
                if match_date >= date: # Current match is not included in the training data
                    pass
                else:
                    match_tensor = torch.from_numpy(row[5:].values)
                    print(match_tensor)


        except Exception as e:
            print(e)
            pass

    
    # Creates the encoder training data
    def createEncoderData(self, batch_size, seq_len, n_features): # Implement removal of unused rows for training data tensor
        # train_data: pandas dataframe
        out_data = torch.zeros((len(self.time_series), seq_len, n_features))
        targets = torch.zeros((len(self.time_series), seq_len, 14))

        # For all matches, try to get the sequence of matches leading to it
        # Load into out_data tensor
        # Load in targets
        # Ensure the size of the tensors is correct
        # Batch the tensors?
        # Return these tensors

        # Adding the sequential data for each match into a tensor. Also creating the target values
        for index, row in self.match_stats.iterrows():
            try:
                if index % 100 == 0:
                    print(index)
                date = row[1]
                team = row[2]
                next_match_stats = torch.from_numpy(row[3:17].to_numpy(dtype = np.float64))
                team_data = self.time_series[self.time_series['Team 1'].str.contains(team)]
                i = 1
                
                for idx, match in team_data.iterrows():
                    match_date = match[1]

                    if match_date < date and i <= seq_len:
                        out_data[index][seq_len - i] = torch.from_numpy(match[5:].to_numpy(dtype = np.float64)) # Automatically pads with zeros, as the data is added in reverse order # Most recent matches at the bottom
                        targets[index][seq_len - i] = next_match_stats
                        next_match_stats = torch.from_numpy(match[7:21].to_numpy(dtype = np.float64))
                        i += 1

            except Exception as e:
                print(e)
                pass

            #print(out_data[index])
        
        # Duplicates the tensor along seq_len dimension to allow masking
        # expanded_tensor = einops.repeat(out_data, 'b m n -> b s m n', s = seq_len) # Correctly duplicates across the seq_len - now just need to mask

        # # Batch up the data
        # n_batches = expanded_tensor.shape[0] // batch_size
        # batched_tensor = expanded_tensor[:n_batches * batch_size]
        # batched_tensor = batched_tensor.reshape(n_batches, batch_size, seq_len, seq_len, n_features)

        # batched_targets = targets[:n_batches * batch_size]
        # batched_targets = batched_targets.reshape(n_batches, batch_size, seq_len, 14)

        torch.save(out_data, 'encoder_training_data.pt')
        torch.save(targets, 'encoder_targets.pt')
        # pytorch tensor of the time series data batched into shape (n_batches, batch_size, seq_len, seq_len, 14) and target tensor     (14 being the number of target stats for a given match)


    def createPretrainingData(self, seq_len, n_features):
        out_data = torch.zeros((len(self.match_results), seq_len, n_features * 2))
        targets = torch.zeros((len(self.match_results), 28))
        n = 0 # Number of matches used in the created training data

        for index, row in self.match_results.iterrows():
            print(index)
            date = row[1]
            team1 = row[2]
            team2 = row[3]
            result = row[4]

            team1_data = self.time_series[self.time_series['Team 1'].str.contains(team1)]
            team2_data = self.time_series[self.time_series['Team 1'].str.contains(team2)]

            try:
                target_list = self.findMatchStats(team1, date) + self.findMatchStats(team1, date)
                target = torch.tensor(target_list)

                team1_tensor = torch.zeros((seq_len, n_features))
                team2_tensor = torch.zeros((seq_len, n_features))

                i1 = 1
                i2 = 1
                for (idx1, match1), (idx2, match2) in zip(team1_data.iterrows(), team2_data.iterrows()):
                    match1_date = match1[1]
                    match2_date = match2[1]

                    if match1_date < date and i1 <= seq_len:
                        team1_tensor[seq_len - i1] = torch.from_numpy(match1[5:].to_numpy(dtype = np.float64))
                        i1 += 1
                        #print('adding', torch.from_numpy(match1[5:].to_numpy(dtype = np.float64)))
                    
                    if match2_date < date and i2 <= seq_len:
                        team2_tensor[seq_len - i2] = torch.from_numpy(match2[5:].to_numpy(dtype = np.float64))
                        i2 += 1

                if i1 <= 3 or i2 <= 3: # At least three matches need to be in each of the sequences
                    raise

                concat_tensor = torch.concat((team1_tensor, team2_tensor), dim = -1)
                # Expected concat shape: (seq_len, 88)
                #print('Concat shape:', concat_tensor.shape)

                out_data[n] = concat_tensor
                targets[n] = target
                n += 1

            except Exception as e:
                print(e)
                pass

        print('n:', n)
        out_data = out_data[0:n] # Remove unused rows from tensor
        targets = targets[:n]
        print('out_data:', out_data.shape)

        torch.save(out_data, 'pretraining_data.pt')
        torch.save(targets, 'pretraining_targets.pt')

    def createTrainingData(self, seq_len, n_features):
        out_data = torch.zeros((len(self.match_results), seq_len, n_features * 2))
        targets = torch.zeros((len(self.match_results), 3))
        n = 0 # Number of matches used in the created training data

        for index, row in self.match_results.iterrows():
            print(index)
            date = row[1]
            team1 = row[2]
            team2 = row[3]
            result = row[4]
            result_array = [0, 0, 0]

            team1_data = self.time_series[self.time_series['Team 1'].str.contains(team1)]
            team2_data = self.time_series[self.time_series['Team 1'].str.contains(team2)]

            try:
                result_array[result] += 1
                target = torch.tensor(result_array)

                team1_tensor = torch.zeros((seq_len, n_features))
                team2_tensor = torch.zeros((seq_len, n_features))

                i1 = 1
                i2 = 1
                for (idx1, match1), (idx2, match2) in zip(team1_data.iterrows(), team2_data.iterrows()):
                    match1_date = match1[1]
                    match2_date = match2[1]

                    if match1_date < date and i1 <= seq_len:
                        team1_tensor[seq_len - i1] = torch.from_numpy(match1[5:].to_numpy(dtype = np.float64))
                        i1 += 1
                        #print('adding', torch.from_numpy(match1[5:].to_numpy(dtype = np.float64)))
                    
                    if match2_date < date and i2 <= seq_len:
                        team2_tensor[seq_len - i2] = torch.from_numpy(match2[5:].to_numpy(dtype = np.float64))
                        i2 += 1

                if i1 <= 3 or i2 <= 3: # At least three matches need to be in each of the sequences
                    raise

                concat_tensor = torch.concat((team1_tensor, team2_tensor), dim = -1)
                # Expected concat shape: (seq_len, 88)
                #print('Concat shape:', concat_tensor.shape)

                out_data[n] = concat_tensor
                targets[n] = target
                n += 1

            except Exception as e:
                print(e)
                pass

        print('n:', n)
        out_data = out_data[:n] # Remove unused rows from tensor
        targets = targets[:n]
        print('out_data:', out_data.shape)

        torch.save(out_data, 'training_data.pt')
        torch.save(targets, 'training_targets.pt')

    def findTimeSeries(self, team1, team2, date, seq_len = 12, n_features = 44):
        team1_data = self.time_series[self.time_series['Team 1'].str.contains(team1)]
        team2_data = self.time_series[self.time_series['Team 1'].str.contains(team2)]

        try:
            team1_tensor = torch.zeros((seq_len, n_features))
            team2_tensor = torch.zeros((seq_len, n_features))

            i1 = 1
            i2 = 1
            for (idx1, match1), (idx2, match2) in zip(team1_data.iterrows(), team2_data.iterrows()):
                match1_date = match1[1]
                match2_date = match2[1]

                if match1_date < date and i1 <= seq_len:
                    team1_tensor[seq_len - i1] = torch.from_numpy(match1[5:].to_numpy(dtype = np.float64))
                    i1 += 1
                    #print('adding', torch.from_numpy(match1[5:].to_numpy(dtype = np.float64)))
                
                if match2_date < date and i2 <= seq_len:
                    team2_tensor[seq_len - i2] = torch.from_numpy(match2[5:].to_numpy(dtype = np.float64))
                    i2 += 1

            if i1 <= 3 or i2 <= 3: # At least three matches need to be in each of the sequences
                raise

            concat_tensor = torch.concat((team1_tensor, team2_tensor), dim = -1)
            # concat shape: (seq_len, 88)

            return concat_tensor.reshape(1, seq_len, n_features * 2) # Adds batch dimension

        except Exception as e:
            print('Not enough data available for this match')
            return None

if __name__ == '__main__':
    start_time = time.time()
    data = Data()
    #data.getData()
    #data.createTrainingData()
    #data.createPretrainData()
    #data.updateData()
    #data.createTimeSeriesData()
    #data.createEncoderData(batch_size = 100, seq_len = 12, n_features = 44)
    data.createPretrainingData(12, 44)
    data.createTrainingData(12, 44)
    data.updateData()
    print('Finished in %s seconds' % round(time.time() - start_time, 2))