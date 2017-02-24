import numpy as np
import theano.tensor as T
import pylab
from PIL import Image
from theano.tensor.nnet import conv
import theano


rng = np.random.RandomState(23455)
input = T.tensor4(name='image', dtype='float64')
w_shap = (2, 3, 9, 9)
w_bound = np.sqrt(3*9*9)
W = theano.shared(
    np.asarray(
        rng.uniform(
            low=-1.0/w_bound,
            high=1.0/w_bound,
            size=w_shap
        ), dtype=input.dtype), name='w')
b_shap = (2,)
b = theano.shared(np.asarray(
    rng.uniform(
        low=-.5, high=.5, size=b_shap
    ), dtype=input.dtype), name='b')
cov_out = conv.conv2d(input, W)
output = T.nnet.sigmoid(cov_out + b.dimshuffle('x', 0, 'x', 'x'))
f = theano.function([input], output)

img_file = open('3wolfmoon.jpg', 'rb')
img = Image.open(img_file)
img = np.asarray(img, dtype='float64')/256
cc = img.transpose(2, 0, 1)
img_ = img.transpose(2, 0, 1).reshape(1, 3, 639, 516)
filtered_img = f(img_)
pylab.subplot(1, 3, 1); pylab.axis('off'); pylab.imshow(img)
pylab.gray();
pylab.subplot(1, 3, 2); pylab.axis('off'); pylab.imshow(filtered_img[0, 0, :, :])
pylab.subplot(1, 3, 3); pylab.axis('off'); pylab.imshow(filtered_img[0, 1, :, :])
pylab.show()





