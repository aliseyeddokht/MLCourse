import numpy as np
from matplotlib import pyplot as plt

from unsupervised.kmeans import KMeans

ds = np.loadtxt("../../datasets/kmeans")
kms = KMeans(4, ds)

clusters, costs = kms.start()

i = 0
for c in clusters:
    c = np.array(c)
    plt.plot(c[:, 0], c[:, 1], "o", label=f"cluster-{i}")
    i += 1
plt.legend()
plt.show()

kms.visualize_model_performance()
