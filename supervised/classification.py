import matplotlib.pyplot as plt
import numpy as np

from supervised.model import Model


class Classification(Model):
    def __init__(self, theta_shape, learning_rate, converge_tolerance, converge_metric, max_iterations,
                 penalty_coefficient):
        super().__init__(theta_shape, learning_rate, converge_tolerance, converge_metric, max_iterations,
                         penalty_coefficient)

    def evaluate(self, X_train, y_train, X_val, y_val):
        y_h_train = self.predict(X_train)
        TP_train = np.sum(np.logical_and(y_train == 1, y_h_train >= 0.5))
        TN_train = np.sum(np.logical_and(y_train == 0, y_h_train < 0.5))
        FP_train = np.sum(np.logical_and(y_train == 0, y_h_train >= 0.5))
        FN_train = np.sum(np.logical_and(y_train == 1, y_h_train < 0.5))

        y_h_val = self.predict(X_val)
        TP_val = np.sum(np.logical_and(y_val == 1, y_h_val >= 0.5))
        TN_val = np.sum(np.logical_and(y_val == 0, y_h_val < 0.5))
        FP_val = np.sum(np.logical_and(y_val == 0, y_h_val >= 0.5))
        FN_val = np.sum(np.logical_and(y_val == 1, y_h_val < 0.5))

        metrics = {}
        metrics["ACCURACY_TRAIN"] = (TP_train + TN_train) / (TP_train + TN_train + FP_train + FN_train)
        metrics["ACCURACY_VAL"] = (TP_val + TN_val) / (TP_val + TN_val + FP_val + FN_val)
        metrics["SPECIFICITY_TRAIN"] = TN_train / (TN_train + FP_train)
        metrics["SPECIFICITY_VAL"] = TN_val / (TN_val + FP_val)
        metrics["PRECISION_TRAIN"] = TP_train / (TP_train + FP_train)
        metrics["PRECISION_VAL"] = TP_val / (TP_val + FP_val)
        metrics["RECALL_TRAIN"] = TP_train / (TP_train + FN_train)
        metrics["RECALL_VAL"] = TP_val / (TP_val + FN_val)
        metrics["J_TRAIN"] = self.J(y_train, y_h_train)
        metrics["J_VAL"] = self.J(y_val, y_h_val)

        return metrics

    def visualize_model_performance(self):
        metrics = self.iterations_metrics
        plt.title("Classification Evaluation")
        plt.xlabel("Iteration")
        plt.ylabel("Metric")

        iterations, accuracy = range(len(metrics["ACCURACY_TRAIN"])), metrics["ACCURACY_TRAIN"]
        plt.plot(iterations, accuracy, label="Accuracy_Training")
        iterations, accuracy = range(len(metrics["ACCURACY_VAL"])), metrics["ACCURACY_VAL"]
        plt.plot(iterations, accuracy, label="Accuracy_Validation")

        iterations, specificity = range(len(metrics["SPECIFICITY_TRAIN"])), metrics["SPECIFICITY_TRAIN"]
        plt.plot(iterations, specificity, label="Specificity_Training")
        iterations, specificity = range(len(metrics["SPECIFICITY_VAL"])), metrics["SPECIFICITY_VAL"]
        plt.plot(iterations, specificity, label="Specificity_Validation")

        iterations, precision = range(len(metrics["PRECISION_TRAIN"])), metrics["PRECISION_TRAIN"]
        plt.plot(iterations, precision, label="Precision_Training")
        iterations, precision = range(len(metrics["PRECISION_VAL"])), metrics["PRECISION_VAL"]
        plt.plot(iterations, precision, label="Precision_Validation")

        iterations, recall = range(len(metrics["RECALL_TRAIN"])), metrics["RECALL_TRAIN"]
        plt.plot(iterations, recall, label="Recall_Training")
        iterations, recall = range(len(metrics["RECALL_VAL"])), metrics["RECALL_VAL"]
        plt.plot(iterations, recall, label="Recall_Validation")

        plt.legend()
        plt.show()

        iterations, J = range(len(metrics["J_TRAIN"])), metrics["J_TRAIN"]
        plt.plot(iterations, J, label="J_Training")
        iterations, J = range(len(metrics["J_VAL"])), metrics["J_VAL"]
        plt.plot(iterations, J, label="J_Validation")

        plt.legend()
        plt.show()


class LinearLogisticRegression(Classification):
    def __init__(self, number_of_features, learning_rate=1e-8, converge_tolerance=100, converge_metric="ACCURACY_TRAIN",
                 max_iterations=1000, penalty_coefficient=0):
        super().__init__((number_of_features, 1), learning_rate, converge_tolerance, converge_metric, max_iterations,
                         penalty_coefficient)

    def J(self, y, y_h):
        penalty = self.penalty_coefficient / 2 * np.linalg.norm(self.theta) ** 2
        return np.sum(-(y * np.log(y_h)) - ((1 - y) * np.log(1 - y_h))) + penalty

    def predict(self, X):
        return 1 / (1 + np.exp(-X @ self.theta))

    def gradient(self, X, y):
        penalty = self.penalty_coefficient * self.theta
        y_h = self.predict(X)
        return X.T @ (y_h - y) + penalty


class PolynomialLogisticRegression(Classification):
    def __init__(self, degree, number_of_features, learning_rate=1e-8, converge_tolerance=100,
                 converge_metric="ACCURACY_TRAIN", max_iterations=1000, penalty_coefficient=0):
        super().__init__((number_of_features, degree + 1), learning_rate, converge_tolerance, converge_metric,
                         max_iterations, penalty_coefficient)

    def J(self, y, y_h):
        penalty = self.penalty_coefficient / 2 * np.linalg.norm(self.theta) ** 2
        return np.sum(-(y * np.log(y_h)) - ((1 - y) * np.log(1 - y_h))) + penalty

    def predict(self, X):
        N = len(X)
        K, D = self.theta.shape
        Tensor = np.repeat(X, D).reshape((N, K, D)) ** range(D) * self.theta
        result = np.sum(np.sum(Tensor, axis=1), axis=1)
        result = 1 / (1 + np.exp(-result))
        return np.expand_dims(result, axis=1)

    def gradient(self, X, y):
        penalty = self.penalty_coefficient * self.theta
        N = len(X)
        K, D = self.theta.shape
        Tensor = np.repeat(X, D).reshape((N, K, D)) ** range(D)
        M = K * D
        error = np.repeat((self.predict(X) - y), M).reshape((N, K, D))
        return np.sum(Tensor * error, axis=0) + penalty
