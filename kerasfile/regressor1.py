import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import load_boston

def minmax_normalization(data):
    xs_max = np.max(data, axis=0)
    xs_min = np.min(data, axis=0)
    xs = (1 - 0)*(data - xs_min)/(xs_max - xs_min) - 0
    return xs

X_data = load_boston().data
X_data = minmax_normalization(X_data)
Y_data = load_boston().target[:,np.newaxis]

x_train, y_train = X_data[:400], Y_data[:400]
x_test, y_test = X_data[400:450], Y_data[400:450]

model = Sequential()
model.add(Dense(output_dim=1, input_dim=13))
model.compile(loss='mse', optimizer = 'sgd')

print('train----------')
for i in range(1001):
    cost = model.train_on_batch(x_train, y_train)
    if i%50 == 0:
        print('train cost:', cost)

print('test-----------')
cost = model.evaluate(x_test, y_test,batch_size=50)
print('cost:',cost)

y_pred = model.predict(x_test)
print(y_pred)
print(y_test)



