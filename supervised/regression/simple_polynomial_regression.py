import matplotlib.pyplot as plt
import numpy as np

import visualizer
from polynomial_regression import PolynomialRegression

# number of training examples
N = 100

# number of validation examples
M = N // 4

X_train = np.random.uniform(-10, 10, N)
noise = np.random.normal(0, 3, N)
y_train = X_train ** 2 + noise

X_val = np.random.uniform(-15, 15, M)
noise = np.random.normal(0, 3, M)
y_val = X_val ** 2 + noise

X_train = np.vstack((np.ones_like(X_train), X_train)).T
X_val = np.vstack((np.ones_like(X_val), X_val)).T
y_train = np.expand_dims(y_train, axis=1)
y_val = np.expand_dims(y_val, axis=1)

regressor = PolynomialRegression(2, 2, learning_rate=1e-7)
metrics = regressor.fit(X_train, y_train, X_val, y_val)

plt.title("Simple Polynomial Regression")
plt.xlabel("X")
plt.ylabel("Y")
plt.plot(X_train[:, 1], y_train, "o", label="Training")
plt.plot(X_val[:, 1], y_val, "o", label="Validation")
X = np.linspace(plt.xlim()[0], plt.xlim()[1], M + N)
X = np.vstack((np.ones_like(X), X)).T
y = regressor.hypothesis_function(X)
plt.plot(X[:, 1], y, "k", label="Regression-Curve")
plt.legend()
plt.show()

visualizer.show(metrics, plt)
