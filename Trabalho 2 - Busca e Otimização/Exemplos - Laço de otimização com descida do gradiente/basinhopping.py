from scipy.optimize import basinhopping
import numpy as np
import math
import matplotlib.pyplot as plt

# Function for minimization (squared)
def squared(x):
    return np.square(x)

# Function for minimization (Rastrigin)
def rastrigin(x):
    A = 10
    return A + sum([(X**2 - A * np.cos(2 * math.pi * X)) for X in x])

xlimits = (-5.12 , 5.12)
f = rastrigin
x0 = [4]

minimizer_kwargs = {"method": "BFGS"}
res = basinhopping(f, x0, minimizer_kwargs=minimizer_kwargs, niter=200)

print(res)

print('Sucess: {}'.format(res.lowest_optimization_result.success))
print('Pos: {}'.format(res.lowest_optimization_result.x))
print('Min: {}'.format(res.lowest_optimization_result.fun))

x = np.arange(xlimits[0], xlimits[1], 0.1)
y = [f([X]) for X in x]

plt.plot(x, y)
plt.plot(res.lowest_optimization_result.x, res.lowest_optimization_result.fun, 'ro')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

