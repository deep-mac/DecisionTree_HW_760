import numpy as np
from scipy.interpolate import lagrange
import math
from numpy.polynomial.polynomial import Polynomial

x = np.random.uniform(-math.pi, math.pi, 100)
e = np.random.uniform(0, 0.0, 100)
print (x)

y = np.sin(x)
print (y)

p = lagrange(x+e, y)

print (p)


x_test = np.random.uniform(-math.pi, math.pi, 100)
y_truth = np.sin(x)

y_pred = p(x_test)

mse = np.square(np.subtract(y_truth, y_pred)).mean()

print("Printing mse = ", mse)
