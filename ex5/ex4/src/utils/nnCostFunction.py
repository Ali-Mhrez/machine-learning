# -*- coding: utf-8 -*-
import numpy as np

from .sigmoid import sigmoid
from .sigmoidGradient import sigmoidGradient


def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda): 
    """
   nnCostFunction Implements the neural network cost function for a two layer
   neural network which performs classification
   nnCostFunction computes the cost and gradient of the neural network. The
   parameters for the neural network are "unrolled" into the vector
   nn_params and need to be converted back into the weight matrices.

   The returned parameter grad should be a "unrolled" vector of the
   partial derivatives of the neural network.
    """

# Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
# for our 2 layer neural network
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                       (hidden_layer_size, input_layer_size + 1), order='F').copy()

    Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):],
                       (num_labels, (hidden_layer_size + 1)), order='F').copy()

# Setup some useful variables
    m = X.shape[0]
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)

    """
    ====================== YOUR CODE HERE ======================
    Instructions: You should complete the code by working through the
                  following parts.
    
    Part 1: Feedforward the neural network and return the cost in the
            variable J. After implementing Part 1, you can verify that your
            cost function computation is correct by verifying the cost
            computed in ex4.py
    Part 2: Implement the backpropagation algorithm to compute the gradients
            Theta1_grad and Theta2_grad. You should return the partial derivatives of
            the cost function with respect to Theta1 and Theta2 in Theta1_grad and
            Theta2_grad, respectively. After implementing Part 2, you can check
            that your implementation is correct by running checkNNGradients
            
    Note:   The vector y passed into the function is a vector of labels
            containing values from 1..K. You need to map this vector into a 
            binary vector of 1's and 0's to be used with the neural network
            cost function.
    
    Hint:   We recommend implementing backpropagation using a for-loop
            over the training examples if you are implementing it for the 
            first time.
    
    Part 3: Implement regularization with the cost function and gradients.
    
    Hint:   You can implement this around the code for backpropagation. 
            That is, you can compute the gradients for the regularization 
            separately and then add them to Theta1_grad and Theta2_grad from Part 2.
    """

    X = np.c_[np.ones(m), X]
    out1 = X.dot(Theta1.T)          # (m,l1)
    out1_act = sigmoid(out1)

    out1_act = np.c_[np.ones(m), out1_act]
    out2_act = sigmoid(out1_act.dot(Theta2.T))    # (m,l2)

    one_hot = np.eye(num_labels)[y-1]
    
    J = (-1/m) * np.sum(one_hot * np.log(out2_act) + (1-one_hot) * np.log(1 - out2_act))
    J += (Lambda/(2*m)) * np.sum(Theta1[:, 1:]**2) + (Lambda/(2*m)) * np.sum(Theta2[:, 1:]**2)
    
    errors = out2_act - one_hot     # (m,l2)
    prev_errors = errors.dot(Theta2[:,1:]) * sigmoidGradient(out1)  # (m,l1)
    
    Theta2_grad = (1/m) * errors.T.dot(out1_act)
    Theta2_grad[:, 1:] += (Lambda/m) * (Theta2[:, 1:])
    
    Theta1_grad = (1/m) * prev_errors.T.dot(X)
    Theta1_grad[:, 1:] += (Lambda/m) * (Theta1[:, 1:])
    # =========================================================================

    # Unroll gradient
    grad = np.hstack((Theta1_grad.T.ravel(), Theta2_grad.T.ravel()))

    return J, grad