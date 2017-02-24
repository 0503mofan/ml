from sklearn.datasets import load_digits
import numpy as np
from sklearn.learning_curve import learning_curve
from sklearn.svm import SVC
import matplotlib.pyplot as plt


data = load_digits()
X_data = data.data
y_data = data.target
model = SVC(gamma=0.01)
train_sizes, train_loss, test_loss = learning_curve(model, X_data, y_data, cv=10, scoring='neg_mean_squared_error', train_sizes=[0.1, 0.25, 0.5, 0.75, 1.0])
if __name__ == '__main__':
    train_loss_mean = -np.mean(train_loss, axis=1)
    test_loss_mean = -np.mean(test_loss, axis=1)

plt.plot(train_sizes, train_loss_mean, 'o-', color='r', label='Training')
plt.plot(train_sizes, test_loss_mean, 'o-', color='g', label='Testing')
plt.show()
