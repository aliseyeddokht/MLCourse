import matplotlib.pyplot as plt

import datasets.data_loader
from supervised.regression import LinearRegression

dataset_file = "../../datasets/simple_linear_regression"
X_train, y_train, X_val, y_val, K = datasets.data_loader.load(dataset_file)

regressor = LinearRegression(K, learning_rate=1e-6)
regressor.fit(X_train, y_train, X_val, y_val)
predict = regressor.predict(X_val)

plt.title("Simple Linear Regression")
plt.xlabel("$X$")
plt.ylabel("$Y$")
plt.plot(X_train[:, 1], y_train, "o", label="Training")
plt.plot(X_val[:, 1], y_val, "o", label="Validation")
plt.plot(X_val[:, 1], predict, "k", label="Regression-Line")
plt.legend()
plt.show()

regressor.visualize_model_performance()
