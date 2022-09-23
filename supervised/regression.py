import matplotlib.pyplot as plt
import numpy as np

from supervised.model import Model


class Regression(Model):
    def __init__(self, theta_shape, learning_rate, converge_tolerance, converge_metric, max_iterations):
        super().__init__(theta_shape, learning_rate, converge_tolerance, converge_metric, max_iterations)

    def J(self, y, y_h):
        e = y_h - y
        return np.asscalar(0.5 * e.T @ e)

    def evaluate(self, X_train, y_train, X_val, y_val):
        y_train_h = self.predict(X_train)
        error_train = y_train_h - y_train
        y_val_h = self.predict(X_val)
        error_val = y_val_h - y_val
        metrics = {}
        metrics["MSE_TRAIN"] = np.mean(error_train ** 2)
        metrics["MSE_VAL"] = np.mean(error_val ** 2)
        metrics["RMSE_TRAIN"] = np.sqrt(metrics["MSE_TRAIN"])
        metrics["RMSE_VAL"] = np.sqrt(metrics["MSE_VAL"])
        metrics["MAE_TRAIN"] = np.mean(np.abs(error_train))
        metrics["MAE_VAL"] = np.mean(np.abs(error_val))
        metrics["J_TRAIN"] = self.J(y_train, y_train_h)
        metrics["J_VAL"] = self.J(y_val, y_val_h)

        return metrics

    def visualize_model_performance(self):
        metrics = self.iterations_metrics
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
        plt.suptitle(f"Regression Evaluation")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("MSE")
        iterations, mse = range(len(metrics["MSE_TRAIN"])), metrics["MSE_TRAIN"]
        ax1.plot(iterations, mse, label="Training")
        iterations, mse = range(len(metrics["MSE_VAL"])), metrics["MSE_VAL"]
        ax1.plot(iterations, mse, label="Validation")
        ax1.legend()

        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("RMSE")
        iterations, rmse = range(len(metrics["RMSE_TRAIN"])), metrics["RMSE_TRAIN"]
        ax2.plot(iterations, rmse, label="Training")
        iterations, rmse = range(len(metrics["RMSE_VAL"])), metrics["RMSE_VAL"]
        ax2.plot(iterations, rmse, label="Validation")
        ax2.legend()

        ax3.set_xlabel("Iteration")
        ax3.set_ylabel("MAE")
        iterations, mae = range(len(metrics["MAE_TRAIN"])), metrics["MAE_TRAIN"]
        ax3.plot(iterations, mae, label="Training")
        iterations, mae = range(len(metrics["MAE_VAL"])), metrics["MAE_VAL"]
        ax3.plot(iterations, mae, label="Validation")
        ax3.legend()

        ax4.set_xlabel("Iteration")
        ax4.set_ylabel("J")
        iterations, J = range(len(metrics["J_TRAIN"])), metrics["J_TRAIN"]
        ax4.plot(iterations, J, label="Training")
        iterations, J = range(len(metrics["J_VAL"])), metrics["J_VAL"]
        ax4.plot(iterations, J, label="Validation")
        ax4.legend()

        plt.show()




class LinearRegression(Regression):
    def __init__(self, number_of_features, learning_rate=1e-8, converge_tolerance=100, converge_metric="RMSE_TRAIN",
                 max_iterations=1000):
        super().__init__((number_of_features, 1), learning_rate, converge_tolerance, converge_metric, max_iterations)

    def predict(self, X):
        return X @ self.theta

    def gradient(self, X, y):
        y_h = self.predict(X)
        return X.T @ (y_h - y)


class NormalEquationRegression:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.theta = np.linalg.pinv(X.T @ X) @ X.T @ y

    def predict(self, X):
        return X @ self.theta


class PolynomialRegression(Regression):
    def __init__(self, degree, number_of_features, learning_rate=1e-8, converge_tolerance=100,
                 converge_metric="RMSE_TRAIN", max_iterations=1000):
        super().__init__((number_of_features, degree + 1), learning_rate, converge_tolerance, converge_metric,
                         max_iterations)

    def predict(self, X):
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
        error = np.repeat((self.predict(X) - y), M).reshape((N, K, D))
        return np.sum(Tensor * error, axis=0)
