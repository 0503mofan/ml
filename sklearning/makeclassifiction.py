from sklearn.datasets import make_classification
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt


X,y = make_classification(n_samples=500, n_features=2, n_redundant=0, n_informative=2, random_state=22, n_clusters_per_class=1, scale=100)
X = preprocessing.scale(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
model = SVC()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))

