from sklearn import datasets
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing

boston = datasets.load_boston()
X_data = boston.data
y_data = boston.target
X_data = preprocessing.scale(X_data)
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3)
model = LinearRegression(normalize=True)
model.fit(X_train, y_train)
prediction = model.predict(X_test)
#print(prediction)
#print(model.coef_)
#print(model.intercept_)
print(model.get_params)
print(model.score(X_test, y_test))#R*2 coefficient of determination(regression)