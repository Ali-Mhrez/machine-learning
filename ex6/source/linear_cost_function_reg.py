import numpy as np

def linear_cost_function_reg(theta, X, y, lambda_):
    """
    LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
    regression with multiple variables
      [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
      cost of using theta as the parameter for linear regression to fit the
      data points in X and y. Returns the cost in J and the gradient in grad

    """
    m = X.shape[0]
    cost=0
    gradient = np.zeros(theta.shape)
    """====================== YOUR CODE HERE ======================
    Instructions: Compute the cost and gradient of regularized linear 
                  regression for a particular choice of theta.
    
                  You should set J to the cost and grad to the gradient.
    
    """

    preds = X.dot(theta)
    errors = preds - y
    cost = (1/(2*m)) * errors.dot(errors)
    cost +=  (lambda_/(2*m)) * np.sum(theta[1:]**2)
    
    gradient = (1/m) * X.T.dot(errors)
    gradient[1:] += (lambda_/m) * theta[1:]
    
    return cost, gradient