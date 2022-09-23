import numpy as np


def max_min_normalizer(X):
    return (X - np.min(X)) / (np.max(X) - np.min(X))


def z_score_normalizer(X):
    return (X - X.mean()) / X.std()

