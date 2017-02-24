from sklearn import datasets
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing


iris = datasets.load_iris()
X_data = iris.data
y_data = iris.target
X_data = preprocessing.scale(X_data)
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

print(knn.score(X_test, y_test))
