import jax
import jax.scipy.optimize
import jax.numpy as jnp
from scipy.optimize import minimize
from optimizers import sgd, rmsprop, adam

# Objective function
f = lambda x: jnp.sum(x**2)

# Jacobian by automatic differentiation with Jax
j = jax.jacfwd(f)

# Initial solution
x0 = jnp.array([4.0, 7.0])

res = jax.scipy.optimize.minimize(f, x0=x0, method='BFGS')
print('==========================================')
print('Jax Minimize:\n', res)

res = minimize(f, jac=j, x0=x0, method="BFGS")
print('==========================================')
print('Scipy Minimize:\n', res)

res = sgd(f, x0, j)
print('==========================================')
print('Stochastic Gradient Descent:\n', res)

res = rmsprop(f, x0, j)
print('==========================================')
print('Root Mean Square Propagation:\n', res)

res = adam(f, x0, j, maxiter=2000)
print('==========================================')
print('Adam Stochastic Optimization:\n', res)