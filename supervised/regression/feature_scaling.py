import matplotlib.pyplot as plt
import numpy as np

N = 200

miu = 50,
sigma = 128
X = np.random.normal(miu, sigma, (N, 2))

# z-score normalization
NX = (X - X.mean()) / X.std()

# max-min normalization
MMX = (X - np.min(X)) / (np.max(X) - np.min(X))

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
plt.suptitle("2D Normal Dataset")
ax1.set_title("Actual Data")
ax1.set_xlabel("$X_1$")
ax1.set_ylabel("$X_2$")
ax1.plot(X[:, 0], X[:, 1], "o")
ax2.set_title("Z-Score Normalization")
ax2.set_xlabel("$X_1$")
ax2.set_ylabel("$X_2$")
ax2.plot(NX[:, 0], NX[:, 1], "o")
ax3.set_title("Max-Min Normalization")
ax3.set_xlabel("$X_1$")
ax3.set_ylabel("$X_2$")
ax3.plot(MMX[:, 0], MMX[:, 1], "o")
plt.show()
