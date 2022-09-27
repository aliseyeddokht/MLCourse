import abc

import numpy as np


class Model:
    def __init__(self, theta_shape, learning_rate, converge_tolerance, converge_metric, max_iterations,
                 penalty_coefficient):
        self.learning_rate = learning_rate
        self.converge_tolerance = converge_tolerance
        self.converge_metric = converge_metric
        self.theta = np.random.uniform(0, 1, theta_shape)
        self.max_iterations = max_iterations
        self.iterations_metrics = {}
        self.penalty_coefficient = penalty_coefficient

    @abc.abstractmethod
    def J(self, y, y_h):
        pass

    @abc.abstractmethod
    def evaluate(self, X_train, y_train, X_val, y_val):
        pass

    @abc.abstractmethod
    def predict(self, X):
        pass

    @abc.abstractmethod
    def gradient(self, X, y):
        pass

    @abc.abstractmethod
    def visualize_model_performance(self):
        pass

    def append(self, metrics):
        for key in metrics:
            if key not in self.iterations_metrics:
                self.iterations_metrics[key] = []
            value = metrics[key]
            self.iterations_metrics[key].append(value)

    def fit(self, X_train, y_train, X_val, y_val):
        for i in range(self.max_iterations):
            gradient = self.gradient(X_train, y_train)
            self.theta = self.theta - self.learning_rate * gradient
            metrics = self.evaluate(X_train, y_train, X_val, y_val)
            self.append(metrics)
