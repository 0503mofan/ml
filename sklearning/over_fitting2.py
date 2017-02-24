from sklearn.datasets import load_digits
from sklearn.learning_curve import validation_curve
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt


data = load_digits()
X_data = data.data
y_data = data.target
param_range = np.logspace(-6, -2.3, 5)
train_loss, test_loss = validation_curve(SVC(), X_data, y_data, cv=10, param_name='gamma', param_range=param_range,
                                         scoring='neg_mean_squared_error')
train_loss_mean = -np.mean(train_loss, axis=1)
test_loss_mean = -np.mean(test_loss, axis=1)

plt.plot(param_range, train_loss_mean, 'o-', color='r', label='Training')
plt.plot(param_range, test_loss_mean, 'o-', color='g', label='Testing')
plt.xlabel('gamma')
plt.ylabel('loss')
plt.show()
