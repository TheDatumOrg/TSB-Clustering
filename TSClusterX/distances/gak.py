import numpy as np
from scipy.spatial.distance import cdist
from distances.distance import DistanceMeasure


class GAKDistance(DistanceMeasure):
    def gak_dist(self, x, y, gamma): 
        x, y = np.expand_dims(x, axis=0), np.expand_dims(y, axis=0)
        K = np.exp(-(cdist(x, y, "sqeuclidean") / (2 * gamma ** 2) + np.log(2 - np.exp(cdist(x, y, "sqeuclidean") / (2 * gamma ** 2)))))

        csum = np.zeros((len(x)+1, len(y)+1))
        csum[0][0] = 1
        for i in range(len(x)):
            for j in range(len(y)):
                csum[i+1][j+1] = (csum[i, j + 1] + csum[i + 1, j] + csum[i, j]) * K[i][j]

        return csum[len(x)][len(y)]
    
    def compute(self, series_set):
        n = len(series_set)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                matrix[i, j] = self.gak_dist(series_set[i], series_set[j], gamma=0.1)
        return matrix
