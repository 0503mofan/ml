import pickle, gzip, numpy

f = gzip.open('mnist.pkl.gz', 'rb')
train_set, test_set = pickle.load(f, encoding='bytes')
f.close()
X_train, y_train = train_set
print(X_train[0], y_train[0])