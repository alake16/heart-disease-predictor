import numpy as np
import pandas as pd
import sys
from scipy import stats
from scipy.spatial.distance import euclidean
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

class KNN(object):
    def __init__(self, k=5):
        self.k = k

    def predict(self, train, test, num_predictions, classes):
        predictions = np.zeros(num_predictions)
        row = 0

        for testRow in test:
            closeness = 0
            neighbors = np.full(self.k, sys.maxsize) 
            neighborClasses = np.full(self.k, 0) 

            for trainRow in train:
                closenss = euclidean(trainRow, testRow)
                if(closeness < np.amax(neighbors)):
                    neighbors[np.argmax(neighbors)] = closeness
                    neighborClasses[np.argmax(neighbors)] = classes[row]
            predictions[row] = stats.mode(neighborClasses, axis=None)[0]
            row += 1

        return predictions

def main():
	print('===== Fetching and cleaning up data =====')
	# Get data
	data = pd.read_csv('heart.csv', header=0, sep=",")
	data = data.astype({'oldpeak': 'int64'})
	data = data.dropna(how='any',axis=0)
	# Clean data
	imr = SimpleImputer(missing_values=0, strategy='mean')
	imr = imr.fit(data)
	data = imr.transform(data.values)

	# Separating attributes of dataset into features and target
	X = data[0:, 0:12]
	y = data[0:, 13]

	# Separating train and test sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .15, random_state=0)

	print('===== KNN Classifier Running =====')
	knn = KNN(k = 10)
	predictions = knn.predict(X_train, X_test, y_test.size, y_train)
	print('Accuracy: %.2f' % accuracy_score(y_test, predictions))

main()
