import matplotlib.pyplot as plt
import numpy as np

# number of training samples
N = 100

# number of validation samples
M = N // 3

X_train = np.random.uniform(-100, 100, N)
noise = np.random.normal(0, 16, N)
y_train = 2 * X_train + noise

X_val = np.random.uniform(-150, 150, M)
noise = np.random.normal(0, 16, M)
y_val = 2 * X_val + noise

X_train = np.vstack((np.ones_like(X_train), X_train)).T
X_val = np.vstack((np.ones_like(X_val), X_val)).T
y_train = np.expand_dims(y_train, axis=1)
y_val = np.expand_dims(y_val, axis=1)

theta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ y_train
predict = X_val @ theta

plt.title("Simple Linear Regression[Normal Equation]")
plt.xlabel("X")
plt.ylabel("Y")
plt.plot(X_train[:, 1], y_train, "o", label="Training")
plt.plot(X_val[:, 1], y_val, "o", label="Validation")
plt.plot(X_val[:, 1], predict, "k", label="Regression-Line")
plt.legend()
plt.show()
