import matplotlib.pyplot as plt
import numpy as np

import visualizer
from polynomial_regression import PolynomialRegression

# number of training samples
N = 600

# number of validation samples
M = N // 3

X_train = np.random.uniform(-1, 1, N)
noise = np.random.normal(0, 0.16, N)
y_train = np.sin(X_train * np.pi) + noise

X_val = np.random.uniform(-1, 1, M)
noise = np.random.normal(0, 0.16, M)
y_val = np.sin(X_val * np.pi) + noise

X_train = np.vstack((np.ones_like(X_train), X_train)).T
X_val = np.vstack((np.ones_like(X_val), X_val)).T
y_train = np.expand_dims(y_train, axis=1)
y_val = np.expand_dims(y_val, axis=1)

regressor = PolynomialRegression(5, 2, learning_rate=1e-6)
metrics = regressor.fit(X_train, y_train, X_val, y_val)

plt.title("Sine Approximation")
plt.xlabel("X")
plt.ylabel("Y")
plt.plot(X_train[:, 1], y_train, "o", label="Training")
plt.plot(X_val[:, 1], y_val, "o", label="Validation")
X = np.linspace(plt.xlim()[0], plt.xlim()[1], M + N)
X = np.vstack((np.ones_like(X), X)).T
y = regressor.hypothesis_function(X)
plt.plot(X[:, 1], y, "k", label=f"Regression-Curve")
plt.legend()
plt.show()

visualizer.show(metrics, plt)
