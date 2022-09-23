import numpy as np


def load(file):
    dataset = np.loadtxt(file)
    np.random.shuffle(dataset)
    N = len(dataset)
    dataset_with_bias = np.column_stack((np.ones((N, 1)), dataset))
    M = dataset_with_bias.shape[1]
    train = dataset_with_bias[:int(N * 0.8)]
    X_train, y_train = train[:, :M - 1], train[:, M - 1]
    val = dataset_with_bias[len(train):]
    X_val, y_val = val[:, :M - 1], val[:, M - 1]
    y_train, y_val = np.expand_dims(y_train, 1), np.expand_dims(y_val, 1)
    return X_train, y_train, X_val, y_val, M - 1
