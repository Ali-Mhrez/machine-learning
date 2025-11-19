def computeCost(X, y, theta):
    # COMPUTECOST Compute cost for linear regression
    #   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
    #   parameter for linear regression to fit the data points in X and y

    # Initialize some useful values
    m = y.shape[0]  # number of training examples

    # You need to return the following variables correctly
    J = 0

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta
    #               You should set J to the cost.
    
    # predictions = X.dot(theta)                 # (m,n) . (n,) --> (m,) 
    # errors = predictions - y                   # (m,) - (m,) --> (m,)
    # J = (1 / (2 * m)) * (errors.dot(errors))   # (m,) . (m,) --> ()
    
    for i in range(X.shape[0]):
        pred = X[i].dot(theta)
        J += (pred - y[i])**2
    J /= (2*m)
        

    # =========================================================================

    return J
