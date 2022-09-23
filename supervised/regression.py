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
        return self.converge_tolerance < N and self.converge_tolerance <= np.sum(np.gradient(metric) == 0)

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


class LinearRegression(Regression):
    def __init__(self, number_of_features, learning_rate=1e-8, converge_tolerance=100, converge_metric="RMSE_TRAIN",
                 max_iterations=1000):
        super().__init__((number_of_features, 1), learning_rate, converge_tolerance, converge_metric, max_iterations)

    def hypothesis_function(self, X):
        return X @ self.theta

    def gradient(self, X, y):
        y_h = self.hypothesis_function(X)
        return X.T @ (y_h - y)


class PolynomialRegression(Regression):
    def __init__(self, degree, number_of_features, learning_rate=1e-8, converge_tolerance=100,
                 converge_metric="RMSE_TRAIN", max_iterations=1000):
        super().__init__((number_of_features, degree + 1), learning_rate, converge_tolerance, converge_metric,
                         max_iterations)

    def hypothesis_function(self, X):
        N = len(X)
        K, D = self.theta.shape
        Tensor = np.repeat(X, D).reshape((N, K, D)) ** range(D) * self.theta
        result = np.sum(np.sum(Tensor, axis=1), axis=1)
        return np.expand_dims(result, axis=1)

    def gradient(self, X, y):
        N = len(X)
        K, D = self.theta.shape
        Tensor = np.repeat(X, D).reshape((N, K, D)) ** range(D)
        M = K * D
        error = np.repeat((self.hypothesis_function(X) - y), M).reshape((N, K, D))
        return np.sum(Tensor * error, axis=0)
