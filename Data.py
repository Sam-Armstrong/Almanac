import bs4, requests, datetime, pandas, time, torch
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
    """
    Class used for scraping and managing the data required for training the prediction model
    """
    
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
            self.time_series = None


    def findMatchResult(self, team_name, date):
        #n_include = 7 # The number of past matches to calculate the team average from

        days_since_match = calculateDaysSince(date)
        team_data = self.match_results[self.match_results['Date'].str.contains(date)]
        match_data1 = team_data[team_data['Team 1'].str.contains(team_name)]
        match_data2 = team_data[team_data['Team 2'].str.contains(team_name)]
        result = torch.zeros((3))

        for index, row in match_data1.iterrows():
            result[row[4]] = 1

        for index, row in match_data2.iterrows():
            result[row[4]] = 1

        return result


    def findMatchStats(self, team_name, date):
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
    

    def findTeamStats(self, team_name, date, n_include = 5):
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
        i = 0

        for index, row in team_data.iterrows():
            match_date = row[1]
            days_since = calculateDaysSince(match_date)

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
    

    def updateScrapeData(self):
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

        for i, url in enumerate(url_list):
            print(i + 1, '/', url_list_len)
            
            date = dates[i]
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
                team1_stats.append(att_shots2)
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
                team2_stats.append(att_shots1)
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
        result.to_csv('MatchResults.csv') # Saves all the data to a CSV (overwrites existing file)

        for data in all_stats:
            df_len = len(stats_dataframe)
            stats_dataframe.loc[df_len] = data

        frames = [stats_dataframe, self.match_stats]
        result = pandas.concat(frames)
        result.to_csv('MatchStats.csv')

    def scrapeData(self):
        date_list = []
        league_list = ['Premier-League', 'La-Liga', 'Bundesliga', 'Serie-A', 'Ligue-1', 'Major-League-Soccer', 'Championship', 'League-One', 'Primeira-Liga']

        results_dataframe = pandas.DataFrame(columns = ['Date', 'Team 1', 'Team 2', 'Result'])
        stats_dataframe = pandas.DataFrame(columns = ['Date', 'Team 1', 'Goals', 'Goals Against', 'Possession', 'Shots on Target', 'Attempted Shots', 'Shot Accuracy', 'SoT Against',
                                                      'Att Shots Against', 'Saves', 'Save Accuracy', 'Fouls', 'Fouls Against', 'Corners', 'Corners Against', 'Offsides']) # 17
        all_results = []
        all_stats = []

        # Gets the n - 1 previous dates
        for i in range(1, 3000):
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
                team1_stats.append(att_shots2)
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
                team2_stats.append(att_shots1)
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


    # Creates the encoder training data
    def createEncoderDataResultTargets(self, batch_size, seq_len, n_features):
        # train_data: pandas dataframe
        out_data = torch.zeros((len(self.time_series), seq_len, n_features))
        targets = torch.zeros((len(self.time_series), seq_len, 42)) # 28 (targets) + 14 (avg opposition data)

        # Adding the sequential data for each match into a tensor. Also creating the target values
        for index, row in self.match_stats.iterrows():
            try:
                if index % 10 == 0:
                    print(index)
                date = row[1]
                team1 = row[2]
                date_results = self.match_results[self.match_results['Date'].str.contains(date)]
                team2 = date_results[date_results['Team 1'].str.contains(team1)]['Team 2']
                team2 = date_results[date_results['Team 2'].str.contains(team1)]['Team 1'] if team2.empty else team2
                team2 = team2.item()
                next_match_stats = torch.concat((torch.from_numpy(row[3:17].to_numpy(dtype = np.float64)), torch.from_numpy(np.asarray(self.findMatchStats(team2, date)))), dim = -1)
                team_data = self.time_series[self.time_series['Team 1'].str.contains(team1)]
                i = 1
                
                for idx, match in team_data.iterrows():
                    match_date = match[1]
                    team_name = match[2]
                    opp_team = match[3]

                    if match_date < date and i <= seq_len:
                        out_data[index][seq_len - i] = torch.from_numpy(match[5:].to_numpy(dtype = np.float64)) # Automatically pads with zeros, as the data is added in reverse order
                        targets[index][seq_len - i][:28] = next_match_stats
                        targets[index][seq_len - i][28:] = torch.from_numpy(np.asarray(self.findTeamStats(opp_team, match_date)))
                        next_match_stats = torch.from_numpy(match[7:35].to_numpy(dtype = np.float64))
                        i += 1

            except Exception as e:
                pass
        

        torch.save(out_data, 'encoder_training_data.pt')
        torch.save(targets, 'encoder_targets.pt')
        torch.save(new_training, 'encoder_training_data_no_zeros.pt')
        torch.save(new_targets, 'encoder_targets_no_zeros.pt')
        new_training = out_data[out_data.sum(dim = 0) != 0]
        new_targets = targets[targets.sum(dim = 0) != 0]
        print('out_data shape:', out_data.shape)
        print('out_data with zeros removed shape:', new_training.shape)
        print('targets shape:', targets.shape)
        print('targets with zeros removed shape:', new_targets.shape)
        

    # Creates the encoder training data
    def createEncoderData(self, seq_len, n_features):
        out_data = torch.zeros((len(self.time_series), seq_len, n_features))
        targets = torch.zeros((len(self.time_series), seq_len, 28))
        out_index = 0

        # Adding the sequential data for each match into a tensor. Also creating the target values
        for index, row in self.match_stats.iterrows():
            try:
                if index % 100 == 0:
                    print(index)
                
                date = row[1]
                date_tokens = date.split('-')
                year = int(date_tokens[0])
                month = int(date_tokens[1])
                day = int(date_tokens[2])

                if datetime.date(year, month, day) < datetime.date(2021, 6, 1):
                    team = row[2]
                    next_match_stats = torch.from_numpy(row[3:17].to_numpy(dtype = np.float64))
                    team_data = self.time_series[self.time_series['Team 1'].str.contains(team)]
                    i = 1
                    previous_team2_avg = []
                    
                    for idx, match in team_data.iterrows():
                        match_date = match[1]

                        if match_date < date and i <= seq_len:
                            out_data[out_index][seq_len - i] = torch.from_numpy(match[5:].to_numpy(dtype = np.float64)) # Automatically pads with zeros, as the data is added in reverse order # Most recent matches at the bottom
                            targets[out_index][seq_len - i][:14] = next_match_stats # the next match stats for this team (training labels)
                            targets[out_index][seq_len - i][14:] = torch.from_numpy(previous_team2_avg.to_numpy(dtype = np.float64)) # opposition team average stats (used as train data, not labels)
                            next_match_stats = torch.from_numpy(match[7:21].to_numpy(dtype = np.float64))
                            i += 1
                            
                        previous_team2_avg = match[35:]
                    
                    out_index += 1

            except Exception as e:
                print(e)
                pass
        
        out_data = out_data[:out_index + 1]
        targets = targets[:out_index + 1]
        
        print('Number of rows:', out_index)

        torch.save(out_data, 'encoder_training_data.pt')
        torch.save(targets, 'encoder_targets.pt')
        
        print(out_data.shape)
        print(targets.shape)


    def createPretrainingData(self, seq_len, n_features):
        out_data = torch.zeros((len(self.match_results), seq_len, n_features * 2))
        targets = torch.zeros((len(self.match_results), 28))
        n = 0 # Number of matches used in the created training data

        for index, row in self.match_results.iterrows():
            print(index)
            date = row[1]

            date_tokens = date.split('-')
            year = int(date_tokens[0])
            month = int(date_tokens[1])
            day = int(date_tokens[2])

            if datetime.date(year, month, day) < datetime.date(2021, 6, 1):

                team1 = row[2]
                team2 = row[3]
                result = row[4]

                team1_data = self.time_series[self.time_series['Team 1'].str.contains(team1)]
                team2_data = self.time_series[self.time_series['Team 1'].str.contains(team2)]

                try:
                    target_list = self.findMatchStats(team1, date) + self.findMatchStats(team2, date)
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
                        
                        if match2_date < date and i2 <= seq_len:
                            team2_tensor[seq_len - i2] = torch.from_numpy(match2[5:].to_numpy(dtype = np.float64))
                            i2 += 1

                    if i1 <= 3 or i2 <= 3: # At least three matches need to be in each of the sequences
                        raise

                    concat_tensor = torch.concat((team1_tensor, team2_tensor), dim = -1)
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
            
            date_tokens = date.split('-')
            year = int(date_tokens[0])
            month = int(date_tokens[1])
            day = int(date_tokens[2])

            if datetime.date(year, month, day) < datetime.date(2021, 6, 1):

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
                        
                        if match2_date < date and i2 <= seq_len:
                            team2_tensor[seq_len - i2] = torch.from_numpy(match2[5:].to_numpy(dtype = np.float64))
                            i2 += 1

                    if i1 <= 3 or i2 <= 3: # At least three matches need to be in each of the sequences
                        raise

                    concat_tensor = torch.concat((team1_tensor, team2_tensor), dim = -1)

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
        #print(date)

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
                
                if match2_date < date and i2 <= seq_len:
                    team2_tensor[seq_len - i2] = torch.from_numpy(match2[5:].to_numpy(dtype = np.float64))
                    i2 += 1

            if i1 <= 3 or i2 <= 3: # At least three matches need to be in each of the sequences
                raise

            concat_tensor = torch.concat((team1_tensor, team2_tensor), dim = -1)
            # concat shape: (seq_len, 88)

            return concat_tensor.reshape(1, seq_len, n_features * 2) # Adds batch dimension

        except Exception as e:
            return None


    def createTestTensor(self, file_path, offset = 1):
        odds_df = pandas.read_csv(file_path)
        output_tensor = torch.empty((odds_df.shape[0], 12, 88))
        target_tensor = torch.empty((odds_df.shape[0], 4))
        i = 0

        for index, row in odds_df.iterrows():
            try:
                date = row[0 + offset]
                team1 = row[1 + offset]
                team2 = row[2 + offset]
                result = row[3 + offset]
                odds1 = row[4 + offset]
                odds2 = row[5 + offset]
                odds3 = row[6 + offset]

                prediction_data = self.findTimeSeries(team1, team2, date)

                if prediction_data is not None and torch.sum(prediction_data) != 0:
                    output_tensor[i] = prediction_data
                    target_tensor[i, 0] = result
                    target_tensor[i, 1] = odds1
                    target_tensor[i, 2] = odds2
                    target_tensor[i, 3] = odds3
                    i += 1

            except Exception as e:
                print(e)
                pass

        output_tensor = output_tensor[:i, :, :]
        target_tensor = target_tensor[:i, :]

        torch.save(output_tensor, 'test_input_tensor.pt')
        torch.save(target_tensor, 'test_target_tensor.pt')
        print('Saved test tensors')


if __name__ == '__main__':
    start_time = time.time()
    data = Data()
    data.createTestTensor('test_odds.csv')
    print('Finished in %s seconds' % round(time.time() - start_time, 2))