import numpy as np
from distances.distance import DistanceMeasure
from scipy.spatial.distance import pdist, squareform


class RBFDistance(DistanceMeasure):
    def calculate_RBF_distance(self, X, gamma=None):
        X = np.asarray(X)
        n_samples, n_features = X.shape

        if not gamma:
            gamma = 1.0 / n_features

        sq_dists = squareform(pdist(X, 'sqeuclidean'))
        K = np.exp(-gamma * sq_dists)

        return K

    def compute(self, series_set):
        n = len(series_set)
        matrix = self.calculate_RBF_distance(series_set)
        return matrix
