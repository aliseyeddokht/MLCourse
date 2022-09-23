import matplotlib.pyplot as plt
import numpy as np

import datasets.data_loader
from supervised.regression import PolynomialRegression

dataset_file = "../../datasets/cubic_regression"
X_train, y_train, X_val, y_val, K = datasets.data_loader.load(dataset_file)

regressor = PolynomialRegression(3, K, learning_rate=2e-8)
metrics = regressor.fit(X_train, y_train, X_val, y_val)

plt.title("Cubic Regression")
plt.xlabel("X")
plt.ylabel("Y")
plt.plot(X_train[:, 1], y_train, "o", label="Training")
plt.plot(X_val[:, 1], y_val, "o", label="Validation")
X_curve = np.column_stack((np.ones(50), np.linspace(plt.xlim()[0], plt.xlim()[1])))
y_curve = regressor.predict(X_curve)
plt.plot(X_curve[:, 1], y_curve, "k", label="Regression-Curve")
plt.legend()
plt.show()

regressor.visualize_model_performance()
