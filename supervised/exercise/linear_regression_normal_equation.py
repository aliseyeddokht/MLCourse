import matplotlib.pyplot as plt

import datasets.data_loader
from supervised.regression import NormalEquationRegression

dataset_file = "../../datasets/simple_linear_regression"
X_train, y_train, X_val, y_val, K = datasets.data_loader.load(dataset_file)

regressor = NormalEquationRegression(X_train, y_train)
predict = regressor.predict(X_val)

plt.title("Normal Equation")
plt.xlabel("$X$")
plt.ylabel("$Y$")
plt.plot(X_train[:, 1], y_train, "o", label="Training")
plt.plot(X_val[:, 1], y_val, "o", label="Validation")
plt.plot(X_val[:, 1], predict, "k", label="Regression-Line")
plt.legend()
plt.show()
