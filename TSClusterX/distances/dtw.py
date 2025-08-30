import math
from tqdm import tqdm
import numpy as np
from distances.distance import DistanceMeasure


class DTWDistance(DistanceMeasure):
    def dtw_dist(self, x, y, w):
        N = len(x)
        M = len(y)
        if w == None:
            w = max(N, M)

        D = np.full((N+1, M+1), np.inf)
        D[0, 0] = 0
        
        for i in range(1, N+1):
            for j in range(max(1, i-w), min(i+w, M)+1):
                cost = (x[i-1] - y[j-1])**2
                D[i, j] = cost + min(D[i-1,j],D[i-1,j-1],D[i,j-1])

        Dist = math.sqrt(D[N, M])

        return Dist
    
    def compute(self, series_set):
        n = len(series_set)
        matrix = np.zeros((n, n))
        for i in tqdm(range(n)):
            for j in range(n):
                matrix[i, j] = self.dtw_dist(series_set[i], series_set[j], w=int(len(series_set[i])/10))
        return matrix
