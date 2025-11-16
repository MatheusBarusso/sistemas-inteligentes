# Function for minimization
def f(x):
    return x**2

# Derivative of above
def df(x):
    return 2*x

# Approximates derivative by finite differences (forward)
def df_forward(x, dx):
    return (f(x + dx) - f(x))/dx

# Approximates derivative by finite differences (backward)
def df_central(x, dx):
    return (f(x + dx/2) - f(x - dx/2))/dx


tol = 1e-10
dx = 0.1

xant = 4
x = xant

while True:
    x = xant - df(xant)*dx
    print('Pos: {}'.format(x))

    if abs(x - xant) < tol:
        break

    xant = x

print('Min: {}'.format(f(x)))

