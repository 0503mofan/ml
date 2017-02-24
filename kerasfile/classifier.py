import numpy as np
np.random.seed(1337)
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.optimizers import RMSprop
import gzip, pickle


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, test_set = pickle.load(f, encoding='bytes')
f.close()
x_train1, y_train1 = train_set
x_test1, y_test1 = test_set
x_train = x_train1[:1000]
y_train = y_train1[:1000]
x_test = x_test1[:1000]
y_test = y_test1[:1000]
x_train = x_train.reshape(x_train.shape[0], -1)/255.0
x_test = x_test.reshape(x_test.shape[0], -1)/255.0
y_train = np_utils.to_categorical(y_train, nb_classes=10)
y_test = np_utils.to_categorical(y_test, nb_classes=10)

model = Sequential()
model.add(Dense(output_dim=32, input_dim=784))
model.add(Activation('relu'))
model.add(Dense(output_dim=10))
model.add(Activation('softmax'))


rmsprop = RMSprop(lr=0.008, rho=0.9, epsilon=1e-08,decay=0.0)
model.compile(
    optimizer=rmsprop,
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

print('train-----------')
model.fit(x_train, y_train, nb_epoch=20, batch_size=32)
print('test------------')
loss, accuracy = model.evaluate(x_test, y_test)

print('cost:', loss)
print('accuracy:', accuracy)