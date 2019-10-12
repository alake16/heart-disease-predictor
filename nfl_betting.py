import numpy as np
import pandas as pd

team_info = pd.read_csv('data/nfl_teams.csv', encoding = "ISO-8859-1")
stadium_info = pd.read_csv('data/nfl_stadiums.csv', encoding = "ISO-8859-1")
betting_data = pd.read_csv('data/spreadspoke_scores.csv', encoding = "ISO-8859-1")

print(team_info.head())
print(stadium_info.head())
print(betting_data.head())