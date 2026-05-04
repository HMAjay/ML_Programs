# Implement the non-parametric Locally Weighted Regression algorithm in order to fit data points. 
# Select appropriate data set for your experiment and draw graphs.

import numpy as np
import matplotlib.pyplot as plt

# 1. Generate non-linear data
X = np.linspace(0, 10, 100)
y = np.sin(X) + X/2 + np.random.normal(0, 0.1, 100)

def lowess(x_query, X, y, tau):
    # Add bias term (1s) to X
    X_aug = np.c_[np.ones(len(X)), X]
    x_q = [1, x_query]
    # Vectorized Gaussian weights
    weights = np.exp(-((X - x_query)**2) / (2 * tau**2))
    W = np.diag(weights)
    # Weighted Least Squares Formula: theta = (X^T W X)^-1 X^T W y
    theta = np.linalg.pinv(X_aug.T @ W @ X_aug) @ (X_aug.T @ W @ y)
    return x_q @ theta

# 2. Predict and Plot
X_range = np.linspace(0, 10, 100)
for t, c in zip([0.1, 0.5, 2.0], ['red', 'green', 'blue']):
    y_p = [lowess(xq, X, y, t) for xq in X_range]
    plt.plot(X_range, y_p, color=c, label=f'tau={t}')

plt.scatter(X, y, alpha=0.3)
plt.legend(); plt.show()