import gzip, pickle
import numpy as np
from keras.models import Model
from keras.layers import Dense, Input
import matplotlib.pyplot as plt

f = gzip.open('mnist.pkl.gz')
train_set, test_set = pickle.load(f, encoding='bytes')
f.close()
X_train, y_train = train_set
X_test, y_test = test_set
X_train = X_train[:2000]
X_test = X_test[:2000]
X_train = X_train.astype('float32')/255 - 0.5
X_test = X_test.astype('float32')/255 - 0.5
X_train = X_train.reshape((X_train.shape[0], -1))
X_test = X_test.reshape((X_test.shape[0], -1))
print(X_train.shape)
print(X_test.shape)

encoding_dim = 2
input_img = Input(shape=(784,))
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(10, activation='relu')(encoded)
encoded_output = Dense(encoding_dim)(encoded)
decoded = Dense(10, activation='relu')(encoded_output)
decoded = Dense(64, activation='relu')(decoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='tanh')(decoded)

autoencoder = Model(input=input_img, output=decoded)
encoder = Model(input=input_img, output=encoded_output)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_train, X_train,
                nb_epoch=20,
                batch_size=256,
                shuffle=True)
encoded_imgs = encoder.predict(X_test)
plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1])
plt.show()

