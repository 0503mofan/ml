from sklearn.datasets import load_iris
import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn import preprocessing


data = load_iris()
X_data = data.data
y_data = data.target
#X_data = preprocessing.scale(X_data)
k = []
scores = []
for i in range(1, 31):
    knn = KNeighborsClassifier(n_neighbors=i)
    score = cross_val_score(knn, X_data, y_data, cv=5, scoring='accuracy')#if regression scoing='neg_mean_squraed_err',and add'-' under the cross_val_score
    scores.append(score.mean())
    k.append(i)

plt.plot(k, scores)
plt.xlabel('k for knn')
plt.ylabel('knn scores')
plt.show()


