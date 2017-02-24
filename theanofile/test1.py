import numpy as np
import theano.tensor as T
import theano

state = theano.shared(np.array(0, dtype=np.float64), 'state')
inc = T.scalar('inc', dtype=state.dtype)
accumulator = theano.function([inc], state, updates=[(state, state + inc)])

print(state.get_value())
accumulator(1)
print(state.get_value())
accumulator(10)
print(state.get_value())

state.set_value(-1)
accumulator(1)
print(state.get_value())

temp_fun = state*3 + inc
a = T.scalar(dtype=state.dtype)
skip_shared = theano.function([inc, a], temp_fun, givens=[(state, a)])
print(skip_shared(2,3))
print(state.get_value())