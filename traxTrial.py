# pip install trax==1.3.1

import trax.fastmath.numpy as np

# Watch out for assignments which import import trax.fastmath.numpy as np. If you see this line, remember that when calling np you are really calling Trax’s version of numpy that is compatible with JAX.
# As a result of this, where you used to encounter the type numpy.ndarray now you will find the type jax.interpreters.xla.DeviceArray.

import numpy as numpy
from trax import layers as tl
from trax import shapes
from trax import fastmath

!pip list | grep trax

# grad function in trax
trax.math.grad(f)

################### TRAINING WITH GRAD ###########
#forward and backward propagation
y = model(x)
grads = grad(y.forward)(y.weights, x)

# in a loop - update rule
weights -= alpha*grads


class My_class:
	def __init__(y):
		self.x = y
	def __call__(z):
		self.x += z
		print(self.x)


ia=My_class(10)
ia.x = 10


