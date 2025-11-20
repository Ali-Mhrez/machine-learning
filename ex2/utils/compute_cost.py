# COMPUTECOST Compute cost for linear regression
#   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
#   parameter for linear regression to fit the data points in X and y

def compute_cost(X, y, theta):
    # Initialize some useful values
    m = y.size

    # You need to return the following variable correctly
    J = 0

    # ===================== Your Code Here =====================
    # Instructions : Compute the cost of a particular choice of theta.
    #                You should set J to the cost.

    # ==========================================================
    
    # pred = X.dot(theta)
    # error = y - pred
    # J = 1/(2*m) * error.dot(error)
    
    for i in range(m):
        pred = X[i].dot(theta)
        error = y[i] - pred
        J += error**2
    J /= (2*m)
    
    return J
