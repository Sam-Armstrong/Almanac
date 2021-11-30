"""
Author: Sam Armstrong
Date: 2021
"""

import bs4, requests, datetime, pandas, time

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


    def findTeamStats(self, team_name, date):
        n_include = 7 # The number of past matches to calculate the team average from

        days_since_match = calculateDaysSince(date)
        team_data = self.match_stats[self.match_stats['Team 1'].str.contains(team_name)]
        average_data = []

        goals = 0
        goals_against = 0
        pos = 0
        comp_passes = 0
        pass_acc = 0
        shots_on_t = 0
        att_shots = 0
        shot_acc = 0
        att_shots_against = 0
        comp_passes_against = 0
        pass_acc_against = 0
        saves = 0
        save_acc = 0
        fouls = 0
        fouls_against = 0
        corners = 0
        corners_against = 0
        crosses = 0 
        crosses_against = 0
        touches = 0
        touches_against = 0
        tackles = 0
        interceptions = 0
        inter_against = 0
        duels = 0
        duels_against = 0
        clearences = 0
        offsides = 0

        i = 0

        for index, row in team_data.iterrows():
            match_date = row[0]
            print(match_date) ##

            days_since = calculateDaysSince(match_date)

            if days_since_match - 50 > days_since > days_since_match and i <= n_include:
                goals += row[2]
                goals_against += row[3]
                pos += row[4]
                comp_passes += row[5]
                pass_acc += row[6]
                shots_on_t += row[7]
                att_shots += row[8]
                shot_acc += row[9]
                att_shots_against += row[10]
                comp_passes_against += row[11]
                pass_acc_against += row[12]
                saves += row[13]
                save_acc += row[14]
                fouls += row[15]
                fouls_against += row[16]
                corners += row[17]
                corners_against += row[18]
                crosses += row[19]
                crosses_against += row[20]
                touches += row[21]
                touches_against += row[22]
                tackles += row[23]
                interceptions += row[24]
                inter_against += row[25]
                duels += row[26]
                duels_against += row[27]
                clearences += row[28]
                offsides += row[29]

                i += 1

        if i < 7:
            raise Exception()

        goals /= n_include
        goals_against /= n_include
        pos /= n_include
        comp_passes /= n_include
        pass_acc /= n_include
        shots_on_t /= n_include
        att_shots /= n_include
        shot_acc /= n_include
        att_shots_against /= n_include
        comp_passes_against /= n_include
        pass_acc_against /= n_include
        saves /= n_include
        save_acc /= n_include
        fouls /= n_include
        fouls_against /= n_include
        corners /= n_include
        corners_against /= n_include
        crosses /= n_include 
        crosses_against /= n_include
        touches /= n_include
        touches_against /= n_include
        tackles /= n_include
        interceptions /= n_include
        inter_against /= n_include
        duels /= n_include
        duels_against /= n_include
        clearences /= n_include
        offsides /= n_include

        average_data.append(goals)
        average_data.append(goals_against)
        average_data.append(pos)
        average_data.append(comp_passes)
        average_data.append(pass_acc)
        average_data.append(shots_on_t)
        average_data.append(att_shots)
        average_data.append(shot_acc)
        average_data.append(att_shots_against)
        average_data.append(comp_passes_against)
        average_data.append(pass_acc_against)
        average_data.append(saves)
        average_data.append(save_acc)
        average_data.append(fouls)
        average_data.append(fouls_against)
        average_data.append(corners)
        average_data.append(corners_against)
        average_data.append(crosses)
        average_data.append(crosses_against)
        average_data.append(touches)
        average_data.append(touches_against)
        average_data.append(tackles)
        average_data.append(interceptions)
        average_data.append(inter_against)
        average_data.append(duels)
        average_data.append(duels_against)
        average_data.append(clearences)
        average_data.append(offsides)

        return average_data


    def getData(self):
        date_list = []
        league_list = ['Premier-League', 'La-Liga', 'Bundesliga', 'Serie-A', 'Ligue-1', 'Major-League-Soccer']

        results_dataframe = pandas.DataFrame(columns = ['Date', 'Team 1', 'Team 2', 'Result'])
        stats_dataframe = pandas.DataFrame(columns = ['Date', 'Team 1', 'Goals', 'Goals Against', 'Possession', 'Completed Passes', 'Pass Accuracy', 
                                                      'Shots on Target', 'Attempted Shots', 'Shot Accuracy', 'Att Shots Against', 'Comp Passes Against', 
                                                      'Pass Acc Against', 'Saves', 'Save Accuracy', 'Fouls', 'Fouls Against', 'Corners', 'Corners Against', 
                                                      'Crosses', 'Crosses Against', 'Touches', 'Touches Against', 'Tackles', 'Interceptions', 'Interceptions Against', 
                                                      'Duels', 'Duels Against', 'Clearences', 'Offsides']) # 30
        all_results = []
        all_stats = []

        # Gets the n - 1 previous dates
        for i in range(1, 3000): # (1, n)
            date_list.append(str(datetime.date.fromordinal(datetime.date.today().toordinal() - i)))

        print('Scraping Data... (This could take up to a few hours)')
        
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

                if len(numbers) != 20:
                    raise

                pos1 = numbers[0]
                pos2 = numbers[1]
                
                comp_passes1 = numbers[2]
                att_passes1 = numbers[3]
                pass_acc1 = numbers[4]
                pass_acc2 = numbers[5]
                comp_passes2 = numbers[6]
                att_passes2 = numbers[7]

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

                
                numbers = []
                for d in soup.findAll('div', attrs = {'id': 'team_stats_extra'}):
                    for div in d.findAll('div'):
                        text = div.text
                        
                        for word in text.split():
                            if word.isdigit():
                                numbers.append(int(word))

                if len(numbers) != 22:
                    raise

                fouls1 = numbers[0]
                fouls2 = numbers[1]
                corners1 = numbers[2]
                corners2 = numbers[3]
                crosses1 = numbers[4]
                crosses2 = numbers[5]
                touches1 = numbers[6]
                touches2 = numbers[7]
                tackles1 = numbers[8]
                tackles2 = numbers[9]
                interceptions1 = numbers[10]
                interceptions2 = numbers[11]
                aerial_duels1 = numbers[12]
                aerial_duels2 = numbers[13]
                clearences1 = numbers[14]
                clearences2 = numbers[15]
                offsides1 = numbers[16]
                offsides2 = numbers[17]
                goal_kicks1 = numbers[18]
                goal_kicks2 = numbers[19]
                throw_ins1 = numbers[20]
                throw_ins2 = numbers[21]

                team1_stats.append(date)
                team1_stats.append(teams[0])
                team1_stats.append(scores[0])
                team1_stats.append(scores[1])
                team1_stats.append(pos1)
                team1_stats.append(comp_passes1)
                team1_stats.append(pass_acc1)
                team1_stats.append(shots_on_t1)
                team1_stats.append(att_shots1)
                team1_stats.append(shot_acc1)
                team1_stats.append(att_shots2) # Shots against them
                team1_stats.append(comp_passes2)
                team1_stats.append(pass_acc2)
                team1_stats.append(saves1)
                team1_stats.append(save_acc1)
                team1_stats.append(fouls1)
                team1_stats.append(fouls2)
                team1_stats.append(corners1)
                team1_stats.append(corners2)
                team1_stats.append(crosses1)
                team1_stats.append(crosses2)
                team1_stats.append(touches1)
                team1_stats.append(touches2)
                team1_stats.append(tackles1)
                # Not used tackles 2
                team1_stats.append(interceptions1)
                team1_stats.append(interceptions2)
                team1_stats.append(aerial_duels1)
                team1_stats.append(aerial_duels2)
                team1_stats.append(clearences1)
                team1_stats.append(offsides1)

                team2_stats.append(date)
                team2_stats.append(teams[1])
                team2_stats.append(scores[1])
                team2_stats.append(scores[0])
                team2_stats.append(pos2)
                team2_stats.append(comp_passes2)
                team2_stats.append(pass_acc2)
                team2_stats.append(shots_on_t2)
                team2_stats.append(att_shots2)
                team2_stats.append(shot_acc2)
                team2_stats.append(att_shots1) # Shots against them
                team2_stats.append(comp_passes1)
                team2_stats.append(pass_acc1)
                team2_stats.append(saves2)
                team2_stats.append(save_acc2)
                team2_stats.append(fouls2)
                team2_stats.append(fouls1)
                team2_stats.append(corners2)
                team2_stats.append(corners1)
                team2_stats.append(crosses2)
                team2_stats.append(crosses1)
                team2_stats.append(touches2)
                team2_stats.append(touches1)
                team2_stats.append(tackles2)
                team2_stats.append(interceptions2)
                team2_stats.append(interceptions1)
                team2_stats.append(aerial_duels2)
                team2_stats.append(aerial_duels1)
                team2_stats.append(clearences2)
                team2_stats.append(offsides2)

                all_results.append(match_result)
                all_stats.append(team1_stats)
                all_stats.append(team2_stats)
                

            except:
                pass

        for data in all_results:
            df_len = len(results_dataframe)
            results_dataframe.loc[df_len] = data

        results_dataframe.to_csv('MatchResults.csv')

        for data in all_stats:
            # print(len(data))
            # print(stats_dataframe.shape)
            df_len = len(stats_dataframe)
            stats_dataframe.loc[df_len] = data

        stats_dataframe.to_csv('MatchStats.csv')


    def createTrainingData(self):
        all_data = []

        training_data = pandas.DataFrame(columns = ['Home/Away', 'goals1', 'goals_against1', 'pos1', 'comp_passes1', 'pass_acc1', 'shots_on_t1', 'att_shots1', 'shot_acc1', 'att_shots_against1', 
                                                    'comp_passes_against1', 'pass_acc_against1', 'saves1', 'save_acc1', 'fouls1', 'fouls_against1', 'corners1', 'corners_against1', 
                                                    'crosses1', 'crosses_against1',' touches1', 'touches_against1', 'tackles1', 'interceptions1', 'inter_against1', 'duels1', 
                                                    'duels_against1', 'clearences1', 'offsides1', 'goals2', 'goals_against2', 'pos2', 'comp_passes2', 'pass_acc2', 'shots_on_t2', 
                                                    'att_shots2', 'shot_acc2', 'att_shots_against2', 'comp_passes_against2', 'pass_acc_against2', 'saves2', 'save_acc2', 'fouls2', 
                                                    'fouls_against2', 'corners2', 'corners_against2', 'crosses2', 'crosses_against2',' touches2', 'touches_against2', 'tackles2', 
                                                    'interceptions2', 'inter_against2', 'duels2', 'duels_against2', 'clearences2', 'offsides2', 'Win', 'Draw', 'Loss'])

        for index, row in self.match_results.iterrows():
            print(index, '/', len(self.match_results))
            try:
                date = row[0]
                print('Date: ', date)
                days_since_match = calculateDaysSince(date)
                team1 = row[1].rstrip()
                team2 = row[2].rstrip()
                result = row[3]

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

                full_list1 = [0] + team1_useful_data + team2_useful_data + result_array
                full_list2 = [1] + team2_useful_data + team1_useful_data + opposite_array

                all_data.append(full_list1)
                all_data.append(full_list2)

            except:
                pass # Some matches do not have the correct past data to allow them to be included - hence they are passed over with this except

            for data in all_data:
                df_len = len(training_data)
                training_data.loc[df_len] = data

            training_data.to_csv('TrainingData.csv')


if __name__ == '__main__':
    start_time = time.time()
    data = Data()
    data.getData()
    print('Finished in %s seconds' % round(time.time() - start_time, 2))