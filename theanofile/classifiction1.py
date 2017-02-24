import theano.tensor as T
import theano
from sklearn import datasets
import numpy as np
from sklearn.cross_validation import train_test_split


class Layer(object):
    def __init__(self, inputs, in_size, out_size, activation_function):
        self.W = theano.shared(np.random.normal(0, 1, (in_size, out_size)))
        self.b = theano.shared(np.zeros((out_size, 1)) + 0.1)
        self.Wx_plus_b = T.dot(inputs, self.W) + self.b
        self.activation_function = activation_function
        if self.activation_function is None:
            self.outputs = self.Wx_plus_b
        else:
            self.outputs = self.activation_function(self.Wx_plus_b)

def computer_accuracy(y_target, y_predict):
    correct_prediction = np.equal(y_predict, y_target)
    accuracy = np.sum(correct_prediction)/len(correct_prediction)
    return accuracy

iris = datasets.load_iris()
X_data = iris.data
y_data = iris.target
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3)

x = T.dmatrix('x')
y = T.dmatrix('y')

l1 = Layer(x, 4, 10, T.nnet.softmax)
l2 = Layer(l1.outputs, 10, 1, None)
xent = -y*T.log(l2.outputs) - (1 - y)*T.log(1 - l2.outputs)
cost = xent.mean() + 0.01*((l1.W**2).sum() + (l2.W**2).sum())
gW1, gb1, gW2, gb2 = T.grad(cost, [l1.W, l1.b, l2.W, l2.b])
learning_rate = 0.05
train = theano.function(
    inputs=[x, y],
    outputs=cost,
    updates = [(l1.W, l1.W - learning_rate*gW1), (l1.b, l1.b - learning_rate*gb1),
               (l2.W, l2.W - learning_rate*gW2), (l2.b, l2.b - learning_rate*gb2)]
)
predict = theano.function(inputs = [x], outputs = l2.outputs)

for i in range(501):
    err = train(X_train, y_train)
    if i % 50 == 0:
        print('cost', err)
        print('accuracy',computer_accuracy(y_test, predict(X_test)))

print('target y')
print(y_test)
print('predict y')
print(predict(X_test))
