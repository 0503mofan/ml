import gzip, pickle
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam


f = gzip.open('mnist.pkl.gz')
train_set, test_set = pickle.load(f, encoding='bytes')
f.close()

X_train_all, y_train_all = train_set
X_test_all, y_test_all = test_set
X_train = X_train_all[:1000]
y_train = y_train_all[:1000]
X_test = X_test_all[:100]
y_test = y_test_all[:100]

X_train = X_train.reshape(-1, 1, 28, 28)
X_test = X_test.reshape(-1, 1, 28, 28)
y_train = np_utils.to_categorical(y_train, nb_classes=10)
y_test = np_utils.to_categorical(y_test, nb_classes=10)

model = Sequential()
model.add(Convolution2D(
    nb_filter=32,
    nb_row=5,
    nb_col=5,
    border_mode='same',
    input_shape=(1, 28, 28)
))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))
model.add(Convolution2D(
    nb_filter=64,
    nb_row=5,
    nb_col=5,
    border_mode='same',
))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), border_mode='same'))
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))

adam = Adam(lr=1e-4)
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy']
              )
print('Training----------')
model.fit(X_train, y_train, nb_epoch=1, batch_size=32)
print('Testing-----------')
loss, accuracy = model.evaluate(X_test, y_test)

print('\nloss:', loss)
print('\naccuracy:', accuracy)

