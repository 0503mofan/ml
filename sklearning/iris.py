from sklearn import datasets
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.svm import SVC


data = datasets.load_iris()
X_data = data.data
X_data = preprocessing.scale(X_data)
y_data = data.target
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=.3)
model = SVC()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))