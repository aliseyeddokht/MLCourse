import matplotlib.pyplot as plt
import numpy as np

import visualizer
from linear_regression import LinearRegression

# number of training examples
N = 100

# number of validation examples
M = N // 4

X_train = np.random.uniform(-100, 100, N)
noise = np.random.normal(0, 20, N)
y_train = 2 * X_train + noise

X_val = np.random.uniform(-150, 150, M)
noise = np.random.normal(0, 20, M)
y_val = 2 * X_val + noise

X_train = np.vstack((X_train, np.ones_like(X_train))).T
X_val = np.vstack((X_val, np.ones_like(X_val))).T
y_train = np.expand_dims(y_train, axis=1)
y_val = np.expand_dims(y_val, axis=1)

regressor = LinearRegression(2)
metrics = regressor.fit(X_train, y_train, X_val, y_val)
predict = regressor.hypothesis_function(X_val)

plt.title("Simple Linear Regression")
plt.xlabel("$X$")
plt.ylabel("$Y$")
plt.plot(X_train[:, 0], y_train, "o", label="Training")
plt.plot(X_val[:, 0], y_val, "o", label="Validation")
plt.plot(X_val[:, 0], predict, "k", label="Regression-Line")
plt.legend()
plt.show()

visualizer.show(metrics, plt)
