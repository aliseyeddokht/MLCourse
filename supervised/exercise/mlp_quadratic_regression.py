import matplotlib.pyplot as plt
import numpy as np

import datasets.data_loader
from supervised.mlp import Activation
from supervised.mlp import MultilayerPerceptron
from supervised.mlp import NeuralNetLayer

dataset_file = "../../datasets/mlp_quadratic_regression"
X_train, y_train, X_val, y_val, K = datasets.data_loader.load(dataset_file)

activation = Activation()
layers = [
    NeuralNetLayer(K, 1, lambda x: x ** 2, lambda x: 2 * x),
]

mlp = MultilayerPerceptron(layers, max_epochs=200, learning_rate=6e-8, problem_type="regression")
mlp.fit(X_train, y_train, X_val, y_val)
X_test = np.linspace(-10, 10, 100)
X_test = np.column_stack((X_test, X_test))
X_test[:0] = 1
predict = list(map(lambda x: mlp.predict(np.expand_dims(x, axis=0).T).item(), X_test))

plt.title("MLP Quadratic Regression")
plt.xlabel("$X$")
plt.ylabel("$Y$")
plt.plot(X_train[:, 1], y_train, "o", label="Training")
plt.plot(X_val[:, 1], y_val, "o", label="Validation")
plt.plot(X_test[:, 1], predict, "k", label="Regression-Line")
plt.legend()
plt.show()

mlp.visualize_model_performance()
