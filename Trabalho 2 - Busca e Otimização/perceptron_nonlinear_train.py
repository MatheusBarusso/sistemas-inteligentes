# -*- coding: utf-8 -*-
"""
Exemplo de perceptron não-linear para funções lógicas

@author: Prof. Daniel Cavalcanti Jeronymo
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



# Heaviside step function
@np.vectorize
def heaviside(x):
    return 1.0 if x > 0.0 else 0.0

# Relu step (activation) function
@np.vectorize
def relu(x):
    return x * (x > 0)

# helper function for decision boundary plotting
def perceptron(x, w):
    return heaviside(np.sum(np.concatenate((x,[1])) * w))
    #return relu(np.sum(np.concatenate((x,[1])) * w)) # TODO: delete line above (27) and uncomment this one, then change train() function

def xorperceptron3d(x, y, pand, pnand, por):
    y1 = perceptron([x,y], por)
    y2 = perceptron([x,y], pnand)
    return perceptron([y1, y2], pand)

# calculate MSE (mean squared error) for a training data set
def loss(x, y, w):
    mse = 0
    n = len(y)
    
    for input, output in zip(x, y):
        error = xorperceptron3d(input[0], input[1], w[0:3], w[3:6], w[6:]) - output
        mse += math.pow(error, 2)/n
        
    return mse

# stochastic search for values of weights and bias until MSE is zero
def train(x, y):
    mse = 1
    
    while mse != 0:
        w = np.random.normal(0, 1, 9) # hardcoded with 9 weights (3 or, 3 pnand and 3 and)
        mse = loss(x, y, w)

    return w

# Training data for XOR
x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0, 1, 1, 0])

# Train our neural network
w = train(x,y)
pand = w[0:3]
pnand = w[3:6]
por = w[6:]
print(w)

# 3D plot of training data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x[:,0], x[:,1], y, c='b', marker='o')

# Plot decision boundary
xx1 = np.linspace(0, 1, 100)
xx2 = np.linspace(0, 1, 100)
x, y = np.meshgrid(xx1, xx2)
z = np.vectorize(xorperceptron3d, excluded=['pand', 'pnand', 'por'])(x=x, y=y, pand=pand, pnand=pnand, por=por)
zlow = np.copy(z)
zhigh = np.copy(z)
zlow[z<=0] = np.nan
zhigh[z>0] = np.nan
ax.plot_surface(x, y, zlow, color='r', alpha=0.2)
ax.plot_surface(x, y, zhigh, color='b', alpha=0.2)

# Set labels, view and show
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
ax.view_init(90, 0)
plt.show()
