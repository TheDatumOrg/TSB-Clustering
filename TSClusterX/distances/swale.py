from tqdm import tqdm
import numpy as np
from distances.distance import DistanceMeasure


class SWALEDistance(DistanceMeasure):
    def swale_dist(self, x, y, epsilon):
        cur = np.zeros(len(y))
        prev = np.zeros(len(y))
        for i in range(len(x)):
            prev = cur
            cur = np.zeros(len(y))
            minw = 0
            maxw = len(y)-1
            for j in range(int(minw),int(maxw)+1):
                if i + j == 0:
                    cur[j] = 0
                elif i == 0:
                    cur[j] = j * 5
                elif j == minw:
                    cur[j] = i * 5
                else:
                    if (abs(x[i] - y[i]) <= epsilon):
                        cur[j] = prev[j-1] + 1
                    else:
                        cur[j] = min(prev[j], cur[j-1]) + 5
        return cur[len(y)-1]

    def compute(self, series_set):
        n = len(series_set)
        matrix = np.zeros((n, n))
        for i in tqdm(range(n)):
            for j in range(n):
                matrix[i, j] = self.swale_dist(series_set[i], series_set[j], epsilon=0.2)
        return matrix
