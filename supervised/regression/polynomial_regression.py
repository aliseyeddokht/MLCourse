import numpy as np

from regression import Regression


class PolynomialRegression(Regression):
    def __init__(self, degree, number_of_features, learning_rate=1e-8, converge_tolerance=1e-4,
                 converge_metric="RMSE_TRAIN", max_iterations=1000):
        super().__init__((number_of_features, degree + 1), learning_rate, converge_tolerance, converge_metric,
                         max_iterations)

    def hypothesis_function(self, X):
        N = len(X)
        K, D = self.theta.shape
        Tensor = np.repeat(X, D).reshape((N, K, D))
        result = np.sum(Tensor ** range(D) * self.theta, axis=1).sum(axis=1)
        return np.expand_dims(result, axis=1)

    def gradient(self, X, y):
        N = len(X)
        K, D = self.theta.shape
        Tensor = np.repeat(X, D).reshape((N, K, D)) ** range(D)
        M = K * D
        error = np.repeat((self.hypothesis_function(X) - y), M).reshape((N, K, D))
        return np.sum(Tensor * error, axis=0)
