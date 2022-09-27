from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

import datasets.data_loader
from supervised.classification import PolynomialLogisticRegression

dataset_file = "../../datasets/multiclass_logistic_regression"
X_train, y_train, X_val, y_val, K = datasets.data_loader.load(dataset_file)
lr = 8e-8
mat_iter = 10000

C1_train = X_train[np.where(np.any(y_train == 1, axis=1))]
C1_val = X_val[np.where(np.any(y_val == 1, axis=1))]
C2_train = X_train[np.where(np.any(y_train == 2, axis=1))]
C2_val = X_val[np.where(np.any(y_val == 2, axis=1))]
C3_train = X_train[np.where(np.any(y_train == 3, axis=1))]
C3_val = X_val[np.where(np.any(y_val == 3, axis=1))]
C4_train = X_train[np.where(np.any(y_train == 4, axis=1))]
C4_val = X_val[np.where(np.any(y_val == 4, axis=1))]

plt.title("Logistic Regression")
plt.xlabel("$X$")
plt.ylabel("$Y$")
plt.plot(C1_train[:, 1], C1_train[:, 2], "ob", label="Training")
plt.plot(C1_val[:, 1], C1_val[:, 2], "xb", label="Validation")

plt.plot(C2_train[:, 1], C2_train[:, 2], "og", label="Training")
plt.plot(C2_val[:, 1], C2_val[:, 2], "xg", label="Validation")

plt.plot(C3_train[:, 1], C3_train[:, 2], "or", label="Training")
plt.plot(C3_val[:, 1], C3_val[:, 2], "xr", label="Validation")

plt.plot(C4_train[:, 1], C4_train[:, 2], "oy", label="Training")
plt.plot(C4_val[:, 1], C4_val[:, 2], "xy", label="Validation")

y_train_C1 = deepcopy(y_train)
y_train_C1[y_train_C1 != 1] = 0
y_val_C1 = deepcopy(y_val)
y_val_C1[y_val_C1 != 1] = 0
classifier_1 = PolynomialLogisticRegression(1, K, lr, max_iterations=mat_iter, penalty_coefficient=1)
classifier_1.fit(X_train, y_train_C1, X_val, y_val_C1)

y_train_C2 = deepcopy(y_train)
y_train_C2[y_train_C2 != 2] = 0
y_val_C2 = deepcopy(y_val)
y_val_C2[y_val_C2 != 2] = 0
classifier_2 = PolynomialLogisticRegression(1, K, lr, max_iterations=mat_iter, penalty_coefficient=1)
classifier_2.fit(X_train, y_train_C2, X_val, y_val_C2)

y_train_C3 = deepcopy(y_train)
y_train_C3[y_train_C3 != 3] = 0
y_val_C3 = deepcopy(y_val)
y_val_C3[y_val_C3 != 3] = 0
classifier_3 = PolynomialLogisticRegression(1, K, lr, max_iterations=mat_iter, penalty_coefficient=1)
classifier_3.fit(X_train, y_train_C3, X_val, y_val_C3)

y_train_C4 = deepcopy(y_train)
y_train_C4[y_train_C4 != 4] = 0
y_val_C4 = deepcopy(y_val)
y_val_C4[y_val_C4 != 4] = 0
classifier_4 = PolynomialLogisticRegression(1, K, lr, max_iterations=mat_iter, penalty_coefficient=1)
classifier_4.fit(X_train, y_train_C4, X_val, y_val_C4)

h = 0.5
x_min, x_max = plt.xlim()
y_min, y_max = plt.ylim()
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
X_test = np.c_[xx.ravel(), yy.ravel()]
X_test = np.column_stack((np.ones(len(X_test)), X_test))
y_test_1 = classifier_1.predict(X_test)
y_test_2 = classifier_2.predict(X_test)
y_test_3 = classifier_3.predict(X_test)
y_test_4 = classifier_4.predict(X_test)
y_test = np.argmax(np.column_stack((y_test_1, y_test_2, y_test_3, y_test_4)), axis=1) + 1


def label_to_color(l):
    if l == 1:
        return "b"
    if l == 2:
        return "g"
    if l == 3:
        return "r"
    return "y"


color = list(map(label_to_color, y_test))
plt.scatter(X_test[:, 1], X_test[:, 2], c=color, marker=".", alpha=0.2)
plt.legend()
plt.show()
