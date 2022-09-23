import datasets.data_loader
from supervised.regression import LinearRegression

dataset_file = "../../datasets/multiple_linear_regression"
X_train, y_train, X_val, y_val, K = datasets.data_loader.load(dataset_file)

regressor = LinearRegression(K, learning_rate=6e-6)
metrics = regressor.fit(X_train, y_train, X_val, y_val)
predict = regressor.predict(X_val)
regressor.visualize_model_performance()
