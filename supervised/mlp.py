import enum

import numpy as np
from matplotlib import pyplot as plt


class Activation:
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative_sigmoid(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    def tanh(self, x):
        return 2 * self.sigmoid(2 * x) - 1

    def derivative_tanh(self, x):
        return 4 * self.derivative_sigmoid(2 * x)

    def leaky_relu(self, x):
        return np.maximum(0.1 * x, x)

    def derivative_leaky_relu(self, x):
        d = np.copy(x)
        d[d > 0] = 1
        d[d <= 0] = 0.1
        return d

    def linear(self, x):
        return x

    def derivative_linear(self, x):
        return np.ones_like(x)

    def softmax(self, x):
        s = np.exp(x)
        return s / np.sum(s)

    def derivative_softmax(self, x):
        return self.derivative(self.softmax, x)

    def derivative(self, f, x, epsilon=1e-5):
        return (f(x + epsilon) - f(x - epsilon)) / (2 * epsilon)


class Initialization(enum.Enum):
    UniformXavier = 1
    NormalXavier = 2
    UniformHe = 3
    NormalHe = 4


class NeuralNetLayer:
    def __init__(self, number_of_inputs, number_of_outputs, activation_function, derivative_activation_function,
                 initialization=None):
        if initialization == Initialization.UniformXavier:
            a = np.sqrt(6 / (number_of_inputs + number_of_outputs))
            self.weights = np.random.uniform(-a, a, (number_of_outputs, number_of_inputs))
        elif initialization == Initialization.NormalXavier:
            var = np.sqrt(2 / (number_of_inputs + number_of_outputs))
            self.weights = np.random.normal(0, var, (number_of_outputs, number_of_inputs))
        elif initialization == Initialization.UniformHe:
            a = np.sqrt(6 / number_of_inputs)
            self.weights = np.random.uniform(-a, a, (number_of_outputs, number_of_inputs))
        elif initialization == Initialization.NormalHe:
            var = np.sqrt(2 / number_of_inputs)
            self.weights = np.random.normal(0, var, (number_of_outputs, number_of_inputs))
        else:
            self.weights = np.random.rand(number_of_outputs, number_of_inputs)

        self.activation_function = activation_function
        self.derivative_activation_function = derivative_activation_function

    def net_output(self, x, train_mode):
        if train_mode:
            self.input = x
            self.net_input = self.weights @ x
            return self.activation_function(self.net_input)
        return self.activation_function(self.weights @ x)


class MultilayerPerceptron:

    def __init__(self, net_layers, learning_rate=1e-8, max_epochs=100, penalty_coefficient=0,
                 problem_type="classification"):
        self.net_layers = net_layers
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.penalty_coefficient = penalty_coefficient
        self.iterations_metrics = {}
        self.problem_type = problem_type

    def predict(self, x):
        N = len(self.net_layers)
        output = self.net_layers[0].net_output(x, False)
        for i in range(1, N):
            output = self.net_layers[i].net_output(output, False)
        return output

    def forward_propagate(self, x):
        N = len(self.net_layers)
        output = self.net_layers[0].net_output(x, True)
        for i in range(1, N):
            output = self.net_layers[i].net_output(output, True)
        return output

    def backward_propagate(self, error):
        N = len(self.net_layers)
        layer = self.net_layers[N - 1]
        gradient = layer.derivative_activation_function(layer.net_input)
        sensitivity = -2 * np.identity(len(gradient)) * gradient @ error
        layer.weights -= self.learning_rate * sensitivity @ layer.input.T

        for i in range(N - 2, - 1, -1):
            layer = self.net_layers[i]
            gradient = layer.derivative_activation_function(layer.net_input)
            sensitivity = np.identity(len(gradient)) * gradient @ self.net_layers[i + 1].weights.T @ sensitivity
            layer.weights -= self.learning_rate * sensitivity @ layer.input.T

    def J(self, X, y):
        penalty = np.sum(np.array(list(map(lambda l: np.linalg.norm(l.weights), self.net_layers))))
        cost = self.penalty_coefficient / 2 * penalty
        for x, y_r in zip(X, y):
            x = np.expand_dims(x, axis=0).T
            y_r = np.expand_dims(y_r, axis=0).T
            y_h = self.predict(x)
            if self.problem_type == "classification":
                cost += y_r @ np.log(y_h) + (1 - y_r) @ np.log(1 - y_h)
            else:
                cost += np.linalg.norm(y_r - y_h) ** 2
        return np.asscalar(cost / len(X))

    def evaluate(self, X_train, y_train, X_val, y_val):
        metrics = {}
        metrics["J_TRAIN"] = self.J(X_train, y_train)
        metrics["J_VAL"] = self.J(X_val, y_val)
        return metrics

    def visualize_model_performance(self):
        metrics = self.iterations_metrics
        plt.suptitle(f"MLP Evaluation")
        plt.xlabel("Iteration")
        plt.ylabel("J")
        iterations, J = range(len(metrics["J_TRAIN"])), metrics["J_TRAIN"]
        plt.plot(iterations, J, label="Training")
        iterations, mse = range(len(metrics["J_VAL"])), metrics["J_VAL"]
        plt.plot(iterations, mse, label="Validation")
        plt.legend()
        plt.show()

    def append(self, metrics):
        for key in metrics:
            if key not in self.iterations_metrics:
                self.iterations_metrics[key] = []
            value = metrics[key]
            self.iterations_metrics[key].append(value)

    def fit(self, X_train, y_train, X_val, y_val):
        for i in range(self.max_epochs):
            for x, y in zip(X_train, y_train):
                new_x = np.expand_dims(x, axis=0).T
                new_y = np.expand_dims(y, axis=0).T
                error = new_y - self.forward_propagate(new_x)
                self.backward_propagate(error)
            metrics = self.evaluate(X_train, y_train, X_val, y_val)
            self.append(metrics)
