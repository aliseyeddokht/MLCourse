from regression import Regression


class LinearRegression(Regression):
    def __init__(self, number_of_features, learning_rate=1e-6, converge_tolerance=1e-3, converge_metric="RMSE_TRAIN",
                 max_iterations=1000):
        super().__init__((number_of_features, 1), learning_rate, converge_tolerance, converge_metric, max_iterations)

    def hypothesis_function(self, X):
        return X @ self.theta

    def gradient(self, X, y):
        y_h = self.hypothesis_function(X)
        return X.T @ (y_h - y)
