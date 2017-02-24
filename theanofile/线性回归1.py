import numpy as np
import theano.tensor as T
import theano

x = T.dscalar('x')
y = T.dscalar('y')
z = x + y
f = theano.function([x, y], z)

print(f(1, 3))

x = T.dmatrix('x')
y = T.dmatrix('y')
z = x + y
f = theano.function([x, y], z)

print(f(np.arange(12).reshape(3,4),
      10*np.ones((3,4))))

x = T.dmatrix('x')
s = 1/(1 + T.exp(-x))
logistic = theano.function([x], s)
print(logistic([[2,3],[3,4]]))

x, y = T.dmatrices('x', 'y')
diff = x - y
abs_diff = abs(diff)
diff_squared = diff**2
f = theano.function([x, y], [diff, abs_diff, diff_squared])
print(f(np.arange(12).reshape(3,4), 7*np.ones((3,4))))

