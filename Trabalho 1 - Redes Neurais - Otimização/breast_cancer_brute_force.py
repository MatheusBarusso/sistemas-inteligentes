# -*- coding: utf-8 -*-
"""
Exemplo de perceptron linear para funções lógicas

@author: Prof. Daniel Cavalcanti Jeronymo
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from collections import namedtuple
from scipy import optimize

# Heaviside step function
def heaviside(x):
    return 1 if x > 0 else 0

# Linear predictor
def f(x, w, b):
    return np.sum(np.array(x)*np.array(w)) + b

# Classifier
def fw(x, w, b):
    return heaviside(f(x, w, b))

# calculate MSE (mean squared error) for a training data set
def loss(x, y, w, b):
    mse = 0
    n = len(y)
    
    for input, output in zip(x, y):
        error = fw(input, w, b) - output
        mse += math.pow(error, 2)/n
        
    return mse

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_loss(params, X, y):
    d = X.shape[1]
    w = params[:d]
    b = params[d]
    z = X.dot(w) + b
    p = sigmoid(z)
    epsilon = 1e-12
    loss = -np.mean(y * np.log(p + epsilon) + (1 - y) * np.log(1 - p + epsilon))
    diff = (p - y) / X.shape[0]
    gradiente_w = X.T.dot(diff)
    gradiente_b = np.sum(diff)
    gradiente = np.concatenate([gradiente_w, np.array([gradiente_b])])
    return loss, gradiente

def new_train(x, y):
    X = np.array(x, dtype=float)
    yv = np.array(y, dtype=float)
    n, d = X.shape
    inicial = np.zeros(d + 1)
    def obj(params):
        loss, gradiente = logistic_loss(params, X, yv)
        return loss, gradiente
    resultado = optimize.minimize(fun=lambda p: obj(p)[0],
                                  x0 = inicial,
                                  jac=lambda p: obj(p)[1],
                                  method='L-BFGS-B',
                                  options={'maxiter': 1000})
    params_otimiz = resultado.x
    w_otimiz = params_otimiz[:d]
    b_otimiz = params_otimiz[d]

    b_otimiz = float(b_otimiz)
    mse_class = loss(X, yv, w_otimiz, b_otimiz)
    
    return mse_class, w_otimiz, b_otimiz

# Trains a linear classifier
# stochastic search for values of weights and bias until MSE is zero
def train(x, y):
    BestResult = namedtuple('BestResult', ['mse', 'w', 'b'])
    best = BestResult(100000, [0,0], 0)

    mse = 1
    it = 0
    it_max = 10000
    
    while mse != 0 and it < it_max:
        w = np.random.normal(0, 1, len(x[0]))
        b = np.random.normal(0, 1, 1)
        
        mse = loss(x, y, w, b)
        
        if(mse < best.mse):
            best = BestResult(mse, w, b)
            
        it += 1
        
    return best

# Load the breast cancer dataset
# see:
# http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html
# https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic) 
from sklearn.datasets import load_breast_cancer
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
data = load_breast_cancer()

# Test a linear classifier for each attribute in the dataset
for i in range(len(data.data[0])):
    # Input dataset
    X = data.data[:,i:i+1] #0, 7, 20 e 27
    Y = data.target
    
    # Normalize input 
    scaler = MinMaxScaler()
    scaler.fit(X)
    x = scaler.transform(X)
    
    # Labels don't need normalization for this dataset since they are 0 or 1
    y = Y
    
    # Train the linear classifier
    best_mse, w, b = new_train(x, y)
    print('{}: {}'.format(i,best_mse))

# Plot decision boundary
#plt.plot(x, y, 'ro')
#yp =  [perceptron(xi, w, b) for xi in x]
#plt.plot(x, yp)