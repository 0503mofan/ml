import numpy as np
from theano.tensor.signal import downsample
import theano.tensor as T
import theano


input = T.tensor4(name='input',dtype='float64')
maxpool_shape = (2, 2)
pool_output = downsample.max_pool_2d(input, maxpool_shape, ignore_border=True)
f = theano.function([input], pool_output)

invals = np.random.RandomState(1).rand(3, 2, 5, 5)
print('With ignore_border set to True:')
print('invals[0, 0, :, :] =\n', invals[0, 0, :, :])
print('output[0, 0, :, :] =\n', f(invals)[0, 0, :, :])

pool_output = downsample.max_pool_2d(input, maxpool_shape, ignore_border=False)
f = theano.function([input], pool_output)
print('With ignore_border set to Flase:')
print('invals[1, 0, :, :] =\n', invals[1, 0, :, :])
print('output[1, 0, :, :] =\n', f(invals)[1, 0, :, :])