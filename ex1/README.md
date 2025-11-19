# Linear Regression

## Overview
This exercise implements simple linear regression to predict food truck profit based on city population.

## Setup
1. Clone the repository:
   ```
   git clone https://github.com/Ali-Mhrez/machine-learning.git
   cd machine-learning/ex1/
   ```

2. Run the program:
   ```
   python .
   ```

## What You'll Learn
- **plotData.py**: Visualize training data
- **computeCost.py**: Calculate the cost function (MSE)
- **gradientDescent.py**: Optimize parameters using gradient descent

## Dataset
- `ex1data1.txt`: 97 examples of city population vs. food truck profit
- Features: Population (in 10,000s), Profit (in $10,000s)

## Algorithm
Linear regression: `y = θ₀ + θ₁x`
- Uses gradient descent to find optimal θ values
- Minimizes mean squared error cost function