#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 12:58:33 2019

@author: john
"""

import numpy as np
import random


class NeuralNetwork:
    """
    This class defines methods which can be used to create a neural network
    and train it using the stochastic gradient method.
    """

    
    
    def __init__(self, layer_sizes):
        """
        Here, the variable layer_sizes is expected to be a list of positive
        integers defining the number of neurons in each layer of the network.
        The length of the list defines the number of layers. The first element
        in the list defines the size of the input layer, the second number
        gives the size of the first hidden layer, etc. and the last number 
        gives the size output layer.
        
        Matrices of network weights and vectors of biases are then created, at
        random, for the network defined by layer_sizes using a Gaussian number
        generator. These are intended to give the network an initial
        configuration and to be optimised by the training algorithm
        """
        self.num_layers = len(layer_sizes)
        self.layer_sizes= layer_sizes
        
        self.biases = [ np.random.randn(n,1) for n in layer_sizes[1:] ]
        self.weights= [ np.random.randn(n,m)
                        for n, m in zip(layer_sizes[1:], layer_sizes[:-1]) ]
    
    
    
    def set_weights_biases(self, w, b):
        """
        This method can be used to set the weights and biases to predefined
        values. If, for example, a network was trained and its weights and
        biases were saved to a file, this method can be used to recreate
        the trained network from that file.
        """
        self.weights = w
        self.biases = b
        
    
    
    def feed_forward(self, arg):
        """
        This method returns the output vector of the network after it is
        fed the input vector arg.
        """
        for W, b in zip( self.weights, self.biases ):
            arg = sigmoid( W.dot(arg) + b )
        
        return arg
    
    
    
    def back_propagation(self, a, target):
        """
        This method uses back propagation to calculate the gradient of the 
        Euclidean distance between the output vector of the network, after it 
        has been fed the vector 'a', and the desired output vector 'target'. 
        """
        
        '''
        The input vector 'a' is first fed into the network to compute the
        output. While computing the output, relevant terms for calculating
        the gradient are stored in the following lists:
        '''
        activations = [ a ]
        zeds = []
        
        for W, b in zip( self.weights, self.biases ):
            z = W.dot( a ) + b
            a = sigmoid( z )
            
            activations.append( a )
            zeds.append( z )

        # gradient of Euclidean distance w.r.t. to z for the output layer:
        delta = (a - target) * sigmoid_prime(z)
        
        # gradients w.r.t. weight matrices and bias vectors will be stored here
        nWs = [ np.dot(delta, activations[-2].T) ]
        nbs = [ delta ]
        
        '''
        Here we begin propagating backward, or pulling back, the gradient
        vector calculated above to compute the gradient w.r.t. weights and 
        biases associated with previous layers.
        '''
        for W, z, a in zip( reversed(self.weights[1:]), 
                            reversed(zeds[:-1]),
                            reversed(activations[:-2]) ):
            
            delta = W.T.dot(delta) * sigmoid_prime( z )
            
            nWs.append( np.dot(delta, a.T) )
            nbs.append( delta )
        
        '''
        The gradient for the weights and biases associated with the last
        later was calculated and appended to nWs, nbs first and the gradient
        associated with those for the first layer were calculate last. We
        should reverse these lists to match the ordering of the weights and
        biases lists
        '''
        nWs.reverse()
        nbs.reverse()
        
        return (nWs, nbs)
    
    
    
    def update(self, mini_batch, eta):
        """
        This method will update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The mini_batch is a list of tuples (x, y) x being an input vector
        and y being the desired or target output. Eta is the learning rate.
        """
        nabla_W = [np.zeros(W.shape) for W in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        n = len(mini_batch)
        
        for x, y in mini_batch:
            nWs, nbs = self.back_propagation(x,y)
            
            nabla_W = [nW + nWx for nW, nWx in zip(nabla_W, nWs) ]
            nabla_b = [nb + nbx for nb, nbx in zip(nabla_b, nbs) ]
            
        self.weights = [W - (eta/n)*nW 
                        for W, nW in zip( self.weights, nabla_W)]
        self.biases  = [b - (eta/n)*nb 
                        for b, nb in zip( self.biases, nabla_b)]
        
        
        
    def stochastic_gradient_decent(self, training_data, epochs, 
                                   mini_batch_size, eta):
        """
        This method will train the neural network using mini-batch stochastic
        gradient descent.  The argument training_data is a list of tuples
        (x, y) x being an input vector and y being the desired or target 
        output. 
        
        The variable epochs defines the number of epochs to train the network
        for and mini_batch_size defines the size of the batches the training
        data should be divided into. Eta is the learning rate
        """        
        n = len(training_data)
        
        for j in range(epochs):
            
            random.shuffle(training_data)
            
            mini_batches = [training_data[k:k+mini_batch_size]
                            for k in range(0, n, mini_batch_size)]
            
            for mini_batch in mini_batches:
                self.update(mini_batch, eta)
    
    
    
    def get_hessian(self, a, y):
        from itertools import chain
        
        num_param = sum( [w.size for w in chain(self.weights, self.biases)] )
        
        H = np.zeros([num_param, num_param])
        
        # Forward propagation
        activations = [ a ]
        zeds = []
        
        deltas = []
        
        g = [ [None]*(self.num_layers-1) for i in range(self.num_layers-1)]
        B = [ [None]*(self.num_layers-1) for i in range(self.num_layers-1)]
        
        for W, b in zip( self.weights, self.biases ):
            z = W.dot( a ) + b
            a = sigmoid( z )
            
            activations.append( a )
            zeds.append( z )
            
        for p in range(self.num_layers-1):
            
            gqp = np.eye(self.layer_sizes[p+1])
            g[p][p] = gqp
            
            for q in range(p+1, self.num_layers-1):
                gqp = (self.weights[q]*(sigmoid_prime(zeds[q-1]).T)).dot( g[q-1][p] )
                g[q][p] = gqp
                g[p][q] = 0*gqp.T

        # gradient of Euclidean distance w.r.t. to z for the output layer:
        delta = (a - y) * sigmoid_prime(z)
        deltas.insert(0, delta)
        
#        # gradients w.r.t. weight matrices and bias vectors will be stored here
#        nWs = [ np.dot(delta, activations[-2].T) ]
#        nbs = [ delta ]
        
        
        # Back propagation
        for W, z, a in zip( reversed(self.weights[1:]), 
                            reversed(zeds[:-1]),
                            reversed(activations[:-2]) ):
            
            delta = W.T.dot(delta) * sigmoid_prime( z )
            deltas.insert(0, delta)
            
#            nWs.append( np.dot(delta, a.T) )
#            nbs.append( delta )
#        nWs.reverse()
#        nbs.reverse()
            
        HL = np.eye(self.layer_sizes[-1]).dot( np.diag(sigmoid_prime(zeds[-1]).flatten()) )
        HL = np.diag(sigmoid_prime(zeds[-1]).flatten()).dot(HL)
        HL += np.diag(sigmoid_double_prime(zeds[-1]).flatten()).dot( np.diag((activations[-1]-y).flatten()) )
        
        for p in range(self.num_layers-1):
            B[-1][p] = HL.dot( g[-1][p] )
            
        for p in range(self.num_layers-1):
            for q in range(self.num_layers-2,-1,-1):
                B[q][p] = self.weights[q+1].T.dot(B[q+1][p])*sigmoid_prime(zeds[q])
                B[q][p]+= self.weights[q+1].T.dot(deltas[q+1])*sigmoid_double_prime(zeds[q])*g[q][p]
        
        
        # Construct Hessian
        for hc in range(num_param):
            for hr in range(hc, num_param):
                H[hc,hr] = 1
        HH = []
#        temp = [ [None]*(self.num_layers-1) for i in range(self.num_layers-1)]
           
        for p in range(self.num_layers-1):
#            print('p is {0}'.format(p))
            w = self.weights[p]
            b = self.biases[p]
            omega = np.hstack( (w, b) )
            n,m = omega.shape
            enm = np.zeros((n,m))
            
            for i in range(n):
                for j in range(m):
                    enm[i,j] = 1
#                    print(enm)
                    
                    hr = np.asarray([])
                    
                    for q in range(self.num_layers-1):
#                        print('q is {0}'.format(q))
                        hh = np.dot( B[q][p], enm )
                        hh = hh.dot( np.vstack((activations[p],1)) )
                        hh = hh.dot( np.vstack((activations[q],1)).T )
                        
#                        hh+= deltas[q].dot( ( np.vstack( (np.diag( sigmoid_prime(zeds[q-1]).flatten() ) , np.zeros((1,len(zeds[q-1])))) ).dot(g[q-1][p].dot(enm).dot(np.vstack((activations[p],1)))) ).T )
#                        hh+= deltas[q].dot( 
#                                ( np.vstack( (np.diag(sigmoid_prime(zeds[q-1]).flatten() ), 
#                                              np.zeros((1,len(zeds[q-1])))) ).dot(g[q-1][p]
#                                       .dot(enm).dot(np.vstack((activations[p],1)))) ).T )
                        if q:
                            hh2 = np.vstack( (np.diag(sigmoid_prime(zeds[q-1]).flatten() ), np.zeros((1,len(zeds[q-1])))) )
#                            print(hh2)
                            hh2 = hh2.dot( g[q-1][p] )
                            hh2 = hh2.dot( enm )
                            hh2 = hh2.dot( np.vstack((activations[p],1)) )
                            hh2 = deltas[q].dot( hh2.T )
                        else:
                            hh2 = 0
                        
#                        print(hh.shape)
#                        print(hh2.shape)

#                        print(hh.shape)
                        hr = np.hstack( (hr, (hh+hh2).flatten()) )
#                        print(hr)
                    
#                    print(hr.shape)
                    HH.append(hr)
                    
                    enm[i,j] = 0
                    
        
        return H, g, deltas, HL, B, num_param, activations, np.asarray(HH), zeds
                    
    


# Functions for computing the sigmoid function and its derivative:
def sigmoid(z):
    return 1/( 1 + np.exp(-z) )


def sigmoid_prime(z):
    sig = sigmoid(z)
    return sig*(1-sig)


def sigmoid_double_prime(z):
    sig = sigmoid(z)
    return sig*(1-sig)*(1-2*sig)


if __name__ == '__main__':
    net = NeuralNetwork([1,2,1])
    
    x = np.asarray([1]).reshape(1,1)
    y = np.asarray([1]).reshape(1,1)
    
    H,g,deltas, HL, B, num, activations, HH, zeds = net.get_hessian(x, y)
    
    ddww11 = sigmoid_double_prime(zeds[1])*zeds[0][0]*zeds[0][0]
    
    ddww2 = sigmoid_double_prime(zeds[1])*zeds[0][0] * net.weights[1] + sigmoid_prime(zeds[1])*np.asarray([1,0]).reshape(1,2)
    ddww1 = np.diag( sigmoid_prime(zeds[0]).flatten() ).dot( np.asarray([x,0]).reshape(2,1) )
    
    ddww = np.dot( ddww2, ddww1 )
    
#    np.dot( (sigmoid_double_prime(zeds[-1]) * net.weights[1][0,0]).reshape([1,2]) , net.weights[1] ).dot( sigmoid_prime(zeds[0]).reshape([2,1]) ) * x \
#         + sigmoid_prime(zeds[1]).dot( np.asarray([1,0]).reshape((1,2)) ).dot( sigmoid(zeds[0]]) ) * x
    
    print(HH.shape)
    
#    for q in range(net.num_layers-1):
#        for p in range(net.num_layers-1):
#            print('The shape of B with q={0} and p={1} is {2}'.format(q,p,B[q][p].shape) )