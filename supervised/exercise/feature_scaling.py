import matplotlib.pyplot as plt
import numpy as np

from supervised.utils import max_min_normalizer
from supervised.utils import z_score_normalizer

X = np.random.normal(100, 50, (100, 2))

ZX = z_score_normalizer(X)
MX = max_min_normalizer(X)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
plt.suptitle("2D Normal Dataset")
ax1.set_title("Actual Data")
ax1.set_xlabel("$X_1$")
ax1.set_ylabel("$X_2$")
ax1.plot(X[:, 0], X[:, 1], "o")
ax2.set_title("Z-Score Normalization")
ax2.set_xlabel("$X_1$")
ax2.set_ylabel("$X_2$")
ax2.plot(ZX[:, 0], ZX[:, 1], "o")
ax3.set_title("Max-Min Normalization")
ax3.set_xlabel("$X_1$")
ax3.set_ylabel("$X_2$")
ax3.plot(MX[:, 0], MX[:, 1], "o")
plt.show()
