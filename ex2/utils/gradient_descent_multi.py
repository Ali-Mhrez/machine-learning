import numpy as np


# GRADIENTDESCENTMULTI Performs gradient descent to learn theta
#   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
#   taking num_iters gradient steps with learning rate alpha
from utils.compute_cost import compute_cost


def gradient_descent_multi(X, y, theta, alpha, num_iters):
    # Initialize some useful values
    m = y.size
    J_history = np.zeros(num_iters)
    # theta_ = np.copy(theta)
    
    for i in range(0, num_iters):
        # ====================== YOUR CODE HERE ======================
        # Instructions: Perform a single gradient step on the parameter vector
        #               theta.
        #
        # Hint: While debugging, it can be useful to print out the values
        #       of the cost function (computeCostMulti) and gradient here.
        
        # predictions = X.dot(theta)
        # errors = predictions - y
        # theta_ -= (alpha/m) * X.T.dot(errors)
        
        temp = [0 for j in range(X.shape[1])]
        for j in range(m):
            pred = X[j].dot(theta)
            error = pred - y[j]
            for k in range(X.shape[1]):
                temp[k] += error * X[j,k]
        
        for j in range(len(theta)):
            theta[j] -= (alpha / m) * temp[j]
            
        J_history[i] = compute_cost(X, y, theta)    # save the cost

    return theta, J_history
