import numpy as np
from ..utils.sigmoid import sigmoid


def predict_one_vs_all(all_theta, X):
    """
    Predict the label for a trained one-vs-all classifier.

    :param all_theta: matrix containing all thetas for each class. The i_th row contains the thetas for class i
    :param X: the training set

    :return: p vector of predictions for each X element
    """

    m = np.size(X, 0)
    num_labels = np.size(all_theta, 0)

    """
    ================================================ YOUR CODE HERE ====================================================
    Instructions: Complete the following code to make predictions using your learned logistic regression
                  parameters (one-vs-all). You should set p to a vector of predictions, p[i] in range(1, num_labels).

    Hint: you can use `np.argmax` to get the index of a given element in an array.
          Example usage when dealing with a vector
          ```
          p = np.argmax(output)  #  returns the index of the output vector's maximum element
          ```
    """
    preds = np.zeros((m, num_labels))
    for i in range(num_labels):
          theta = all_theta[i]
          preds[:, i] = sigmoid(X.dot(theta))
    p = np.argmax(preds, axis=1)
    
    x = np.c_[np.ones((m, 1)), X]
    preds2 = sigmoid(x.dot(all_theta.T))
    assert np.array_equal(preds, preds2), "Predictions do not match!"
#     p = np.argmax(preds, axis=1)

    return p
