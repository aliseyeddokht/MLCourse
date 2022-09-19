import abc

import numpy as np


class Regression:
    def __init__(self, theta_shape, learning_rate, converge_tolerance, converge_metric, max_iterations):
        self.learning_rate = learning_rate
        self.converge_tolerance = converge_tolerance
        self.converge_metric = converge_metric
        self.theta = np.random.uniform(0, 1, theta_shape)
        self.max_iterations = max_iterations
        self.iterations_metrics = {}

    @abc.abstractmethod
    def hypothesis_function(self, X):
        pass

    @abc.abstractmethod
    def gradient(self, X, y):
        pass

    def converged(self, metric):
        N = len(metric)
        return 2 < N and (np.abs(metric[N - 1] - metric[N - 2]) <= self.converge_tolerance)

    def evaluate(self, X_train, y_train, X_val, y_val):
        error_train = self.hypothesis_function(X_train) - y_train
        error_val = self.hypothesis_function(X_val) - y_val
        metrics = {}
        metrics["MSE_TRAIN"] = np.mean(error_train ** 2)
        metrics["MSE_VAL"] = np.mean(error_val ** 2)
        metrics["RMSE_TRAIN"] = np.sqrt(metrics["MSE_TRAIN"])
        metrics["RMSE_VAL"] = np.sqrt(metrics["MSE_VAL"])
        metrics["MAE_TRAIN"] = np.mean(np.abs(error_train))
        metrics["MAE_VAL"] = np.mean(np.abs(error_val))
        return metrics

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
            if self.converged(self.iterations_metrics[self.converge_metric]):
                break
        return self.iterations_metrics
