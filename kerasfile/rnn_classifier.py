import gzip, pickle
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense,Activation
from keras.optimizers import Adam


TIME_SIZE = 28
INPUT_SIZE = 28
BATCH_SIZE = 50
BATCH_INDEX = 0
OUTPUT_SIZE = 10
CELL_SIZE = 50
LR = 0.001

f = gzip.open('mnist.pkl.gz')
train_set, test_set = pickle.load(f, encoding='bytes')
f.close()

X_train, y_train = train_set
X_test, y_test = test_set
X_train = X_train[:2000]
y_train = y_train[:2000]
X_train = X_train.reshape(-1, 28, 28)/255.0
X_test = X_test.reshape(-1, 28, 28)/255.0
y_train = np_utils.to_categorical(y_train, nb_classes=10)
y_test = np_utils.to_categorical(y_test, nb_classes=10)

model = Sequential()
model.add(SimpleRNN(
    batch_input_shape=(BATCH_SIZE, TIME_SIZE, INPUT_SIZE),
    output_dim=CELL_SIZE
))
model.add(Dense(OUTPUT_SIZE))
model.add(Activation('softmax'))

adam = Adam(LR)
model.compile(
    optimizer=adam,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

for step in range(4001):
    X_batch = X_train[BATCH_INDEX:BATCH_INDEX+BATCH_SIZE, :, :]
    y_batch = y_train[BATCH_INDEX:BATCH_INDEX+BATCH_SIZE, :]
    cost = model.train_on_batch(X_batch, y_batch)

    BATCH_INDEX += BATCH_SIZE
    BATCH_INDEX = 0 if BATCH_INDEX >= X_train.shape[0] else BATCH_INDEX

    if step%500 == 0:
        cost, accuracy = model.evaluate(X_test, y_test, batch_size=32)
        print('test cost:', cost, 'test accuracy:', accuracy)
