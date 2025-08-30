from tqdm import tqdm
import numpy as np
from distances.distance import DistanceMeasure


class MSMDistance(DistanceMeasure):
    def msm_dist(self, new, x, y, c):
        if ((x <= new) and (new <= y)) or ((y <= new) and (new <= x)):
            dist = c
        else:
            dist = c + min(abs(new - x), abs(new - y))
        return dist
    
    def calculate_MSM_distance(self, x, y, c=0.5, w=10):
        xlen = len(x)
        ylen = len(y)
        cost = np.full((xlen, ylen), np.inf)

        cost[0][0] = abs(x[0] - y[0])

        for i in range(1,len(x)):
            cost[i][0] = cost[i-1][0] + self.msm_dist(x[i], x[i-1], y[0], c)

        for i in range(1,len(y)):
            cost[0][i] = cost[0][i-1] + self.msm_dist(y[i], x[0], y[i-1], c)

        for i in range(1,xlen):
            for j in range(max(0, int(i-w)), min(ylen, int(i+w))):
                cost[i][j] = min(cost[i-1][j-1] + abs(x[i] - y[j]),
                                cost[i-1][j] + self.msm_dist(x[i], x[i -1], y[j], c),
                                cost[i][j-1] + self.msm_dist(y[j], x[i], y[j-1], c))

        return cost[xlen-1][ylen-1]

    def compute(self, series_set):
        n = len(series_set)
        matrix = np.zeros((n, n))
        for i in tqdm(range(n)):
            for j in range(n):
                matrix[i, j] = self.calculate_MSM_distance(series_set[i], series_set[j], c=0.5, w=int(series_set.shape[1]/5))
        return matrix
