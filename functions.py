# -*- coding: utf-8 -*- 
import numpy as np

def d_tanh(x):
    dx = 1 - np.tanh(x)**2
    return dx
 
def sigmoid(x):
    return 1 / (1 + np.exp(-x))    

def d_sigmoid(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)

def least_square(y, t):
    batch_size = y.shape[0]
    return 0.5 * np.sum((y-t)**2) / batch_size

def d_least_square(y, t):
    batch_size = t.shape[0]
    dy = (y - t) / batch_size
    return dy

# Relu 関数
def relu(x):
    return np.maximum(0, x)

# Relu 関数の導関数
def d_relu(x):
    return np.where( x > 0, 1, 0)

def acc(y, t):     
    acc = 0.0
    acc = float(np.sum(y == t))/len(y)
    return acc

def random_seq(x):
    ans = 0
    sign = []
    for i in range(len(x)):
        if i%2 == 0:
            ans += x[i]
        else:
            ans -= x[i]
        if ans >= 0:
            sign.append(1)
        else:
            sign.append(0)
    return ans, sign