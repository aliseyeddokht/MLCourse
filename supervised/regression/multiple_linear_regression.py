import matplotlib.pyplot as plt
import numpy as np

import visualizer
from linear_regression import LinearRegression

# number of features
D = 10

# number of training examples
N = 1200

# number of validation examples
M = N // 3

X_train = np.random.uniform(-100, 100, (N, D))
noise = np.random.normal(0, 16, N)
y_train = 3 * np.sum(X_train, axis=1) + noise

X_val = np.random.uniform(-150, 150, (M, D))
noise = np.random.normal(0, 16, M)
y_val = 3 * np.sum(X_val, axis=1) + noise

X_train = np.concatenate((np.ones((N, 1)), X_train), axis=1)
X_val = np.concatenate((np.ones((M, 1)), X_val), axis=1)
y_train = np.expand_dims(y_train, axis=1)
y_val = np.expand_dims(y_val, axis=1)

regressor = LinearRegression(D + 1, learning_rate=1e-8)
metrics = regressor.fit(X_train, y_train, X_val, y_val)
predict = regressor.hypothesis_function(X_val)

visualizer.show(metrics, plt)
