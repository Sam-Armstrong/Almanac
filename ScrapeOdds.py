from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time
import datetime
import pandas


def scrape():
    """
    Script for scraping historical football odds from the web
    """
    
    initial_url = 'https://www.oddsportal.com/soccer/england/premier-league/results/#/page/1/'

    url_starters = ['https://www.oddsportal.com/soccer/england/premier-league/results/#/page/', 
                    'https://www.oddsportal.com/soccer/england/championship/results/#/page/',
                    'https://www.oddsportal.com/soccer/england/league-one/results/#/page/',
                    'https://www.oddsportal.com/soccer/germany/bundesliga/results/#/page/',
                    'https://www.oddsportal.com/soccer/spain/laliga/results/#/page/',
                    'https://www.oddsportal.com/soccer/italy/serie-a/results/#/page/',
                    'https://www.oddsportal.com/soccer/france/ligue-1/results/#/page/']

    #driver = webdriver.Chrome(executable_path = 'C:/Users/Samue/Documents/Chromedriver/chromedriver')
    driver = webdriver.Edge(executable_path = 'C:/Users/Samue/Documents/Edgedriver/msedgedriver')
    driver.get(initial_url)

    options = Options()
    options.headless = True
    options.add_argument("--headless")

    driver.implicitly_wait(10)

    # Clicks the accept cookies button
    accept_cookies_button = driver.find_element(by = By.ID, value = 'onetrust-accept-btn-handler')
    accept_cookies_button.click()

    # Opens the options list
    select_button = driver.find_element(by = By.CLASS_NAME, value = 'user-header-fakeselect')
    select_button.click()

    # Sets the odds to decimal
    odds_selection_list = driver.find_element(by = By.CLASS_NAME, value = 'user-header-fakeselect-options')
    odds_selection_links = odds_selection_list.find_element(by = By.TAG_NAME, value = 'a')
    odds_selection_links.click()
    time.sleep(3)

    odds_dataframe = pandas.DataFrame(columns = ['Date', 'Team 1', 'Team 2', 'Result', 'Odds 1', 'Odds 2', 'Odds 3'])

    #print(driver.find_element_by_xpath("/html/body//div[@class='wrap']//div[@id='mother-main']//div[@id='mother']//div[@id='wrap']//div[@id='box-top']//div[@id='box-bottom']//div[@id='main']//div[@id='col-left']//div[@id='col-content']//div[@id='tournamentTable']//table[@class=' table-main']").text)

    current_row = 0
    odds_counter = 0
    current_starter = 0
    ignore_until_next_date = True

    for current_starter, url_starter in enumerate(url_starters):
        # date, team1, team2, result, odds1, odds2, odds3
        data_row = []
        current_date = datetime.datetime.now().date()
        current_page = 1

        while current_date >= datetime.date(2021, 9, 25): # Collect all odds after 25th September 2021
            url = url_starters[current_starter] + str(current_page) + '/'
            print(url)
            driver.get(url)
            time.sleep(2)

            tokens = driver.find_element_by_xpath("/html/body//div[@class='wrap']//div[@id='mother-main']//div[@id='mother']//div[@id='wrap']//div[@id='box-top']//div[@id='box-bottom']//div[@id='main']//div[@id='col-left']//div[@id='col-content']//div[@id='tournamentTable']//table[@class=' table-main']").text.split('\n')

            for token in tokens:
                if '1 X 2 B' in token:
                    if len(token) < 25 and current_date >= datetime.date(2021, 9, 25):
                        date_string = token.replace(' 1 X 2 B\'s', '')
                        current_date = datetime.datetime.strptime(date_string, '%d %b %Y').date()
                        data_row = []
                        data_row.append(str(current_date))
                        ignore_until_next_date = False
                    else:
                        ignore_until_next_date = True

                if not ignore_until_next_date:
                    if '-' in token and ':' in token:
                        sub_tokens = token.split(' ')
                        team1 = ''
                        team2 = ''
                        goals1 = 0
                        goals2 = 0
                        current_team = 1

                        for t in sub_tokens:
                            if ':' not in t and '-' not in t:
                                if current_team == 1:
                                    if len(team1) == 0:
                                        team1 = team1 + t
                                    else:
                                        team1 = team1 + ' ' + t
                                else:
                                    if len(team2) == 0:
                                        team2 = team2 + t
                                    else:
                                        team2 = team2 + ' ' + t
                            
                            elif '-' in t:
                                current_team = 2

                            elif ':' in t and len(t) < 5:
                                goals_tokens = t.split(':')
                                goals1 = int(goals_tokens[0])
                                goals2 = int(goals_tokens[1])

                        if goals1 > goals2:
                            result = 0
                        elif goals1 == goals2:
                            result = 1
                        else:
                            result = 2

                        data_row.append(team1)
                        data_row.append(team2)
                        data_row.append(result)
                    
                    if '.' in token and len(token) < 7:
                        odds_counter += 1
                        data_row.append(float(token))
                        
                        if odds_counter == 3:
                            odds_dataframe.loc[current_row] = data_row
                            odds_counter = 0
                            current_row += 1
                            data_row = []
                            data_row.append(current_date)

            current_page += 1

    odds_dataframe.to_csv('all_odds.csv')
    print(odds_dataframe)

    driver.quit()
    
    
if __name__ == '__main__':
    scrape()