import numpy as np
from distances.distance import DistanceMeasure


class EuclideanDistance(DistanceMeasure):
    def compute(self, series_set):
        n = len(series_set)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                matrix[i, j] = np.linalg.norm(series_set[i] - series_set[j])
        return matrix
