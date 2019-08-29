#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 12:58:33 2019

@author: john
"""

import numpy as np
import random
from itertools import chain


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
        self.num_param = sum( [w.size for w in chain(self.weights, 
                                                     self.biases)] )
    
    
    
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
    
    
    
    def get_hessian(self, a, target_y):
        """
        This method will calculate the Hessian matrix of the neural network, 
        with respect to the weights and biases, using the method outlined in
        'Second-order stagewise backpropagation for Hessian-matrix analyses and
        investigation of negative curvature' by Mizutani and Dreyfus (2008)
        
        The parameter a is an input to the neural network and target_y
        is the desired outcome
        """
        from scipy.linalg import block_diag
        N = self.num_layers
        
        # Allocate memory to store neuron activations y, neuron inputs x,
        # blocks of the Hessian and the F matrices
        ys = [ a ]
        xs = []
        Hs = [ [None]*(N-1) for i in range(N-1)]
        Fs = [ [None]*(N-1) for i in range(N-1)]
        
        
        # Forward propagation to calculate xs and ys
        for W, b in zip( self.weights, self.biases ):
            x = W.dot( a ) + b
            a = sigmoid( x )
            
            ys.append( a )
            xs.append( x )
            
        # xi is the derivative of the loss function w.r.t. the last activation
        # vector yn
        xi = (a - target_y)
        
        # delta is the derivative of the loss function w.r.t. the last input
        # vector xn
        delta = xi * sigmoid_prime(x)
        
        
        # Calculate Z using eqn (5) in Mizutani and Dreyfus (2008)
        # Note Hessian of the quadratic loss function is an Identity matrix
        dsig = np.diag( sigmoid_prime(x).flatten() )
        Z = dsig.T.dot( np.eye(self.layer_sizes[-1]) ).dot( dsig )
        Z += np.diag( (sigmoid_double_prime(x)*xi).flatten() )
        
        
        # Back propagation to compute the blocks of the Hessian
        for s in range(N-2, -1, -1):
            
            # Compute the derivative of x vector associated with layer s+1
            # w.r.t. vector of network params, theta, for layer s
            # (i.e. w.r.t. weigths and biases which act on layer s)
            dx_dtheta = np.vstack([1, ys[s]]).T
            for _ in range(len(ys[s+1])-1):
                dx_dtheta = block_diag(dx_dtheta, np.vstack([1, ys[s]]).T )
                
            # Get s,s, block of the Hessian using eqn (10) in Mizutani (2008)
            Hs[s][s] = dx_dtheta.T.dot( Z ).dot( dx_dtheta )
            
            dsig = np.diag( sigmoid_prime(xs[s-1]).flatten() )
            
            if not s==N-2:
                # Get off diagonal blocks using eqn (11) in Mizutani (2008)
                for r in range(s+1, N-1):
                    Hs[s][r] = dx_dtheta.T.dot( Fs[s+1][r] )
                    Hs[r][s] = Hs[s][r].T
                    
                if s==0:
                    # convert Hessian into numpy array before returning
                    H = np.vstack( [np.hstack(h) for h in Hs] )
                    return H
                
                # Get off diagonal Fs using eqn (9) in Mizutani (2008)
                for r in range(s+1, N-1):
                    Fs[s][r] = dsig.T.dot(self.weights[s].T).dot(Fs[s+1][r])
            
            # Get diagonal Fs using eqn (7) in Mizutani (2008)
            Fs[s][s] = self.weights[s].T.dot( Z ).dot( dx_dtheta )
            
            column_pad = np.zeros((self.layer_sizes[s],1))
            I = np.eye(self.layer_sizes[s])
            
            dtheta = np.hstack([column_pad, delta[0]*I ])
            for i in range(1,len(delta)):
                dtheta = np.hstack([dtheta, column_pad])
                dtheta = np.hstack([dtheta, delta[i]*I])
            
            Fs[s][s]+= dtheta
            Fs[s][s] = dsig.T.dot( Fs[s][s] )
            
            # Update Z using eqn (5) in Mizutani (2008)
            W = self.weights[s]
            dsig = np.diag( sigmoid_prime(xs[s-1]).flatten() )
            
            xi = W.T.dot( delta )
            delta = xi * sigmoid_prime(xs[s-1])
            
            Z = W.T.dot( Z ).dot( W )
            Z = dsig.T.dot( Z ).dot( dsig )
            Z += np.diag( (sigmoid_double_prime(xs[s-1])*xi).flatten() )
        
        # This return statement is reached if the network has no hidden layers
        return np.vstack( [np.hstack(h) for h in Hs] )
                    
    


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
    
    # Unit test for get_Hessian method
    net = NeuralNetwork([3,4,7,2,4])
    x = np.asarray([1,2,3]).reshape(3,1)
    y = np.asarray([1,2,3,4]).reshape(4,1)
    H = net.get_hessian(x, y)