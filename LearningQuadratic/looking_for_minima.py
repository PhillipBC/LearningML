#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 12:57:18 2019

@author: john
"""

import numpy as np
import neural_network as nn
import matplotlib.pyplot as plt



def func(x):
    """The function we want to learn"""
    return x**2



def loss(net, xs, ys):
    """
    The loss function is the average squared Euclidean distance between the 
    network output and the target output of the test data
    """
    dists = [ (y - net.feed_forward(x))**2
              for x,y in zip(xs,ys) ]
    
    return sum(dists)/len(dists)



# %% Generate training data
print('Generate training data')

xs = np.arange(-1,1,0.01)
num_rand_points = len(xs)
ys = func(xs)
training_data = [ (x,y) for x,y in zip(xs, ys) ]



# %% Train network
print('Train network')

# Set up network and domain, number of epochs and array for output
net = nn.NeuralNetwork([1,2,1])
x = np.arange(-1,1,0.01)
num = 1000
y = np.zeros([num+1,len(x)])

# Save output of network before training begins
y[0,:] = np.asarray( list(map(net.feed_forward, x)) ).reshape(len(x))

# Start training and saving the output of the network
for i in range(1,num+1):
    
    net.stochastic_gradient_decent(training_data, 1, 50, 3.0)
    
    y[i,:] = np.asarray( list(map(net.feed_forward, x)) ).reshape(len(x))

weights = net.weights
biases = net.biases



# %% evaluate cost
print('evaluate cost')

from itertools import chain

c = np.zeros( [sum([w.size for w in chain(weights,biases)]) ,len(x)])

k = 0
for param in chain(weights,biases):
    
    n,m = param.shape
    
    for i in range(n):
        for j in range(m):
            
            omega = param[i,j]

            for l,xi in enumerate(x):
                param[i,j] = omega + 10*xi
                c[k,l] = loss(net, xs, ys)
            
            print('calculated change in loss due to varying omega_{0}'
                  .format(k) )
            param[i,j] = omega
            k += 1



# %% plot the graphs  
cs = []
            
plt.figure(1)
for pic in range(c.shape[0]):
    plt.plot(10*x, c[pic,:], label='{0}'.format(pic) )

plt.title(r'Loss of net in vicinity of trained weigths $\omega_i$')
plt.xlabel(r'$\delta \omega_i$')
plt.ylabel(r'Loss$(\omega +\delta \omega_i)$')
plt.legend()

cs.append(c)


# %% save the weights
ws = []
        
ws.append( [ wi for w in chain(weights, biases) for wi in w.reshape(w.size) ] )



# %%  plot the weights
plt.figure(2) 
for wsi in ws:   
    print(wsi)
    plt.plot(wsi)