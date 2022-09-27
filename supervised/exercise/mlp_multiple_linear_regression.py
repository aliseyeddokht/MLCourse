import datasets.data_loader
from supervised.mlp import Activation, NeuralNetLayer, MultilayerPerceptron

dataset_file = "../../datasets/mlp_multiple_linear_regression"
X_train, y_train, X_val, y_val, K = datasets.data_loader.load(dataset_file)

activation = Activation()
layers = [
    NeuralNetLayer(K, 1, activation.linear, activation.derivative_linear),
]

mlp = MultilayerPerceptron(layers, max_epochs=100, learning_rate=6e-6, problem_type="regression")
mlp.fit(X_train, y_train, X_val, y_val)
mlp.visualize_model_performance()
