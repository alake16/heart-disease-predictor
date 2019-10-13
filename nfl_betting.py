import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

def feature_engineering(data):
	newline = '\n'
	print('===== Variables included in data =====')
	print(data.info())
	print(newline)

	print('===== Give an initial idea of the correlation between variables in dataset =====')
	print(data.corr())
	print(newline)

def main():
	team_info = pd.read_csv('data/nfl_teams.csv', encoding = "ISO-8859-1")
	stadium_info = pd.read_csv('data/nfl_stadiums.csv', encoding = "ISO-8859-1")
	betting_data = pd.read_csv('data/spreadspoke_scores.csv')
	feature_engineering(betting_data)

main()