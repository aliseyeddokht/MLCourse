import matplotlib.pyplot as plt
import numpy as np

import datasets.data_loader
from supervised.classification import LinearLogisticRegression

dataset_file = "../../datasets/linear_logistic_regression"
X_train, y_train, X_val, y_val, K = datasets.data_loader.load(dataset_file)

X_train_C0 = X_train[np.where(np.any(y_train == 0, axis=1))]
X_train_C1 = X_train[np.where(np.any(y_train == 1, axis=1))]

X_val_C0 = X_val[np.where(np.any(y_val == 0, axis=1))]
X_val_C1 = X_val[np.where(np.any(y_val == 1, axis=1))]

classifier = LinearLogisticRegression(K, learning_rate=4e-4, penalty_coefficient=1)
classifier.fit(X_train, y_train, X_val, y_val)

plt.title("Logistic Regression")
plt.xlabel("$X$")
plt.ylabel("$Y$")
plt.plot(X_train_C0[:, 1], X_train_C0[:, 2], "ob", label="Training")
plt.plot(X_val_C0[:, 1], X_val_C0[:, 2], "xb", label="Validation")
plt.plot(X_train_C1[:, 1], X_train_C1[:, 2], "og", label="Training")
plt.plot(X_val_C1[:, 1], X_val_C1[:, 2], "xg", label="Validation")

h = 0.5
x_min, x_max = plt.xlim()
y_min, y_max = plt.ylim()
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
X_test = np.c_[xx.ravel(), yy.ravel()]
X_test = np.column_stack((np.ones(len(X_test)), X_test))
y_test = classifier.predict(X_test)
color = list(map(lambda x: "g" if x >= 0.5 else "b", y_test))
plt.scatter(X_test[:, 1], X_test[:, 2], c=color, marker=".", alpha=0.2)
plt.legend()
plt.show()
classifier.visualize_model_performance()
