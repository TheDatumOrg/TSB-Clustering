import time
import numpy as np
from models.model import BaseClusterModel


def PartitioningAroundMedoids(nclusters, D):
    """
    Performs Partitioning Around Medoids (PAM) clustering algorithm.

    Args:
        nclusters (int): The number of clusters to create.
        D (numpy.ndarray): The distance matrix.

    Returns:
        numpy.ndarray: The cluster labels for each data point.
    """
    medoids = np.random.randint(D.shape[0], size=nclusters)
    n = D.shape[0]
    k = len(medoids)
    maxit = 100

    labels = np.argmin(D[medoids, :], axis=0)
    costs = np.min(D[medoids, :], axis=0)

    cost = np.sum(costs)
    last = 0
    it = 0
    while ((last!=medoids) & (it < maxit)).any():
        best_so_far_medoids = medoids
        for i in range(k):
            medoids_aux = medoids.copy()
            for j in range(n):
                medoids_aux[i] = j
                
                labels_aux = np.argmin(D[medoids_aux, :], axis=0)
                costs_aux = np.min(D[medoids_aux, :], axis=0)

                cost_aux = np.sum(costs_aux)
                if cost_aux < cost:
                    best_so_far_medoids = medoids_aux.copy()
                    cost = cost_aux
                    labels = labels_aux

        last = medoids.copy()
        medoids = best_so_far_medoids
        it = it + 1

    return labels


class PAMClusterModel(BaseClusterModel):
    def fit_predict(self, X):
        print(f"Using parameters: {self.params}")
        start_time = time.time()
        
        if self.distance_matrix is not None:
            # Use precomputed distance matrix
            labels = PartitioningAroundMedoids(self.n_clusters, self.distance_matrix)
        else:
            # Compute distance matrix from data
            from scipy.spatial.distance import pdist, squareform
            distances = pdist(X, metric='euclidean')
            distance_matrix = squareform(distances)
            labels = PartitioningAroundMedoids(self.n_clusters, distance_matrix)
        
        elapsed = time.time() - start_time
        return labels, elapsed