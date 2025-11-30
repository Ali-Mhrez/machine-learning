import numpy as np
import scipy.optimize as optimize

def gaussian_kernel(x1, x2, sigma):
  """
    gaussian_kernel returns a radial basis function kernel between x1 and x2

    def gaussian_kernel(x1, x2) returns a gaussian kernel between x1 and x2
    and returns the value in sim
  """

  # Ensure that x1 and x2 are column vectors
  x1 = x1[:]
  x2 = x2[:]
  
  # You need to return the following variables correctly.
  sim = 0

  # ====================== YOUR CODE HERE ======================
  # Instructions: Fill in this function to return the similarity between x1
  #               and x2 computed using a Gaussian kernel with bandwidth
  #               sigma
  if x1.ndim == 1 and x2.ndim == 1:
        # L2 norm squared between x1 and x2
        squared_diff = np.sum((x1 - x2) ** 2)
        exponent = - squared_diff / (2 * sigma ** 2)
        sim = np.exp(exponent)
    
    # Handle the case where x1 and x2 are matrices (Kernel Matrix output expected)
    # This is the case that SVC requires for training.
  else:
    # Compute the squared Euclidean distance matrix between all pairs of rows
    # The formula for ||xi - xj||^2 can be expanded as:
    # ||xi - xj||^2 = ||xi||^2 + ||xj||^2 - 2 * xi^T * xj
    # 1. Compute ||xi||^2 for each row in x1 (vector of squared norms)
    sum_sq_x1 = np.sum(x1**2, axis=1, keepdims=True)
    
    # 2. Compute ||xj||^2 for each row in x2 (vector of squared norms, transposed)
    sum_sq_x2 = np.sum(x2**2, axis=1, keepdims=True).T
    
    # 3. Compute 2 * xi^T * xj (dot product matrix)
    dot_product_matrix = 2 * x1 @ x2.T
    
    # 4. Compute the squared distance matrix: ||xi - xj||^2
    squared_distance_matrix = sum_sq_x1 + sum_sq_x2 - dot_product_matrix
    
    # Compute the Gaussian kernel matrix
    sim = np.exp(- squared_distance_matrix / (2 * sigma ** 2))

  # =============================================================
  return sim