import matplotlib.pyplot as plt
import numpy as np

from supervised.mlp import Activation, NeuralNetLayer, MultilayerPerceptron

C1_X = np.random.multivariate_normal([10, 20, 0], [[10, 0, 0], [0, 10, 0], [0, 0, 10]], 100)
C1_X[:, 0] = 1
C1_y = np.zeros((100, 4)) + np.array([[1, 0, 0, 0]])

C2_X = np.random.multivariate_normal([10, 0, 20], [[10, 0, 0], [0, 10, 0], [0, 0, 10]], 100)
C2_y = np.zeros((100, 4)) + np.array([[0, 1, 0, 0]])

C3_X = np.random.multivariate_normal([10, -20, 0], [[10, 0, 0], [0, 10, 0], [0, 0, 10]], 100)
C3_y = np.zeros((100, 4)) + np.array([[0, 0, 1, 0]])

C4_X = np.random.multivariate_normal([10, 0, -20], [[10, 0, 0], [0, 10, 0], [0, 0, 10]], 100)
C4_y = np.zeros((100, 4)) + np.array([[0, 0, 0, 1]])

X = np.concatenate((C1_X, C2_X, C3_X, C4_X))
y = np.concatenate((C1_y, C2_y, C3_y, C4_y))
X[:, 0] = 1

indices = np.random.randint(0, 400, 320)
X_train = X[indices]
y_train = y[indices]

indices = np.random.randint(0, 400, 100)
X_val = X[indices]
y_val = y[indices]

plt.title("MLP Multiclass Classification")
plt.plot(C1_X[:, 1], C1_X[:, 2], "or", label="C1")
plt.plot(C2_X[:, 1], C2_X[:, 2], "ob", label="C2")
plt.plot(C3_X[:, 1], C3_X[:, 2], "og", label="C3")
plt.plot(C4_X[:, 1], C4_X[:, 2], "oy", label="C4")

activation = Activation()
layers = [
    NeuralNetLayer(3, 4, activation.linear, activation.derivative_linear),
    NeuralNetLayer(4, 4, activation.linear, activation.derivative_linear),
    NeuralNetLayer(4, 4, activation.linear, activation.derivative_linear),
]

mlp = MultilayerPerceptron(layers, max_epochs=200, learning_rate=6e-6, problem_type="regression")
mlp.fit(X_train, y_train, X_val, y_val)

h = 0.5
x_min, x_max = plt.xlim()
y_min, y_max = plt.ylim()
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
X_test = np.c_[xx.ravel(), yy.ravel()]
X_test = np.column_stack((np.ones(len(X_test)), X_test))
y_test = np.array(list(map(lambda x: mlp.predict(np.expand_dims(x, axis=0).T), X_test)))
colors = ["r", "b", "g", "y"]
plt.scatter(X_test[:, 1], X_test[:, 2], c=list(map(lambda y: colors[np.argmax(y)], y_test)), marker=".", alpha=0.5)
plt.legend()
plt.show()
mlp.visualize_model_performance()
