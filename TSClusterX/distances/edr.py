import numpy as np
from distances.distance import DistanceMeasure


class EDRDistance(DistanceMeasure):
    def edr_dist(self, x, y):
        cur = np.full((1, len(y)), -np.inf)
        prev = np.full((1, len(y)), -np.inf)

        for i in range(len(x)):
            m = 0.1
            minw = 0
            maxw = len(y)-1
            prev = cur
            cur = np.full((len(y)), -np.inf)

            for j in range(int(minw), int(maxw)+1):
                if i + j == 0:
                    cur[j] = 0
                elif i == 0:
                    cur[j] = -j
                elif j == 0:
                    cur[j] = -i
                else:
                    if abs(x[i] - y[j]) <= m:
                        s1 = 0
                    else:
                        s1 = -1
                    cur[j] = max(prev[j - 1] + s1, prev[j] - 1, cur[j - 1] - 1)

        return 0 - cur[len(y) - 1]
    
    def compute(self, series_set):
        n = len(series_set)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                matrix[i, j] = self.edr_dist(series_set[i], series_set[j])
        return matrix
