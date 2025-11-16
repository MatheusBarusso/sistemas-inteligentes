from scipy.optimize import minimize
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

xlimits = [(-5.12 , 5.12)]
f = squared
x0 = [4]

res = minimize(f, x0, method='BFGS', bounds=xlimits, tol=1e-10)

print('Sucess: {}'.format(res.success))
print('Pos: {}'.format(res.x))
print('Min: {}'.format(res.fun))

x = np.arange(xlimits[0][0], xlimits[0][1], 0.1)
y = [f([X]) for X in x]

plt.plot(x, y)
plt.plot(res.x, res.fun, 'ro')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

