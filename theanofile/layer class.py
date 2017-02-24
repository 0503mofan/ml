import numpy as np
import theano.tensor as T
import theano
import random


class Layer(object):
    def __init__(self, inputs, in_size, out_size, activation_function):
        self.W = theano.shared(np.random.normal(0, 1, (in_size, out_size)))
        self.b = theano.shared(np.ones((out_size,)) + 0.1)
        self.Wx_plus_b = T.dot(inputs, self.W) + self.b
        self.activation_function = activation_function
        if self.activation_function is None:
            self.outputs = self.Wx_plus_b
        else:
            self.outputs = self.activation_function(self.Wx_plus_b)