import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.simplefilter("ignore")

def main():
	data = pd.read_csv('heart.csv')
	# Split Train and Test sets
	X_train, X_test, y_train, y_test = train_test_split(data.drop('target', 1), data['target'], test_size = .15, random_state=0)
	# Split Train and Dev sets
	X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size = .15, random_state=0)

	print('===== Using default parameters for LogisticRegression model =====')
	lr = LogisticRegression()
	lr.fit(X_train, y_train)
	predictions = lr.predict(X_dev)
	print('Accuracy: %.2f' % accuracy_score(y_dev, predictions))

	print('===== Manipulating default parameters for LogisticRegression model =====')
	lr = LogisticRegression(penalty='l1', C=100)
	lr.fit(X_train, y_train)
	predictions = lr.predict(X_dev)
	print('Accuracy: %.2f' % accuracy_score(y_dev, predictions))

	print('===== Model Performance on Test Data =====')
	predictions = lr.predict(X_test)
	print('Accuracy: %.2f' % accuracy_score(y_test, predictions))

main()