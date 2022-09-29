import matplotlib.pyplot as plt
import numpy as np


class KMeans:
    def __init__(self, K, dataset, max_iterations=100):
        self.K = K
        self.dataset = dataset
        self.max_iterations = max_iterations

    def J(self, clusters, centroids):
        costs = np.zeros((len(clusters)))
        for i in range(len(centroids)):
            costs[i] = np.linalg.norm(centroids[i] - clusters[i])
        return costs

    def visualize_model_performance(self):
        plt.title("Kmeans Evaluation")
        plt.xlabel("Iterations")
        plt.ylabel("Cost")
        i = 0
        for costs in np.array(self.iterations_costs).T:
            plt.plot(range(len(costs)), costs, label=f"cluster-{i}")
            i += 1
        plt.legend()
        plt.show()

    def start(self):
        clusters, costs = self.make_clusters()
        for i in range(1, self.max_iterations):
            new_clusters, new_costs = self.make_clusters()
            if np.linalg.norm(new_costs) < np.linalg.norm(costs):
                clusters = new_clusters
                costs = new_costs
        return clusters, costs

    def make_clusters(self):
        N = len(self.dataset)
        centroids = self.dataset[np.random.randint(0, len(self.dataset), self.K)]
        labels = -np.ones((N,), dtype=int)
        self.iterations_costs = []
        while True:
            converged = True
            clusters = [[centroids[i]] for i in range(self.K)]
            for i in range(N):
                x = self.dataset[i]
                cluster_id = np.argmin(np.linalg.norm(x - centroids, axis=1), axis=0)
                clusters[cluster_id].append(x)
                if cluster_id != labels[i]:
                    labels[i] = cluster_id
                    converged = False
            centroids = np.array(list(map(lambda c: np.average(c, axis=0), clusters)))
            costs = self.J(clusters, centroids)
            self.iterations_costs.append(costs)
            if converged:
                return clusters, costs


ds = np.loadtxt("../datasets/kmeans")
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
