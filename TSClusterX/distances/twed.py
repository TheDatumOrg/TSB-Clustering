from tqdm import tqdm
import numpy as np
from distances.distance import DistanceMeasure


class TWEDDistance(DistanceMeasure):
    def twed_distance(self, x, timesx, y, timesy, lamb=1, nu=0.0001, w=10):
        xlen = len(x)
        ylen = len(y)
        cur = np.full(ylen, np.inf)
        prev = np.full(ylen, np.inf)

        for i in range(0, xlen):
            prev = cur
            cur = np.full(ylen, np.inf)
            minw = max(0, i - w)
            maxw = min(ylen-1, i + w)
            for j in range(minw, maxw+1):
                if i + j == 0:
                    cur[j] = (x[i] - y[j]) **2
                elif i == 0:
                    c1 = (
                        cur[j - 1]
                        + (y[j - 1] - y[j]) **2
                        + nu * (timesy[j] - timesy[j - 1])
                        + lamb
                    )
                    cur[j] = c1
                elif j == 0:
                    c1 = (
                        prev[j]
                        + (x[i - 1] - x[i]) **2
                        + nu * (timesx[i] - timesx[i - 1])
                        + lamb
                    )
                    cur[j] = c1
                else:
                    c1 = (
                        prev[j]
                        +(x[i - 1] - x[i]) **2
                        + nu * (timesx[i] - timesx[i - 1])
                        + lamb
                    )
                    c2 = (
                        cur[j - 1]
                        + (y[j - 1] - y[j])**2
                        + nu * (timesy[j] - timesy[j - 1])
                        + lamb
                    )
                    c3 = (
                        prev[j - 1]
                        + (x[i] - y[j]) ** 2
                        + (x[i - 1]- y[j - 1]) ** 2
                        + nu
                        * (abs(timesx[i] - timesy[j]) + abs(timesx[i - 1] - timesy[j - 1]))
                    )
                    cur[j] = min(c1, c2, c3)

        return cur[ylen - 1]
    
    def compute(self, series_set):
        n = len(series_set)
        matrix = np.zeros((n, n))
        for i in tqdm(range(n)):
            for j in range(n):
                matrix[i, j] = self.twed_distance(series_set[i], np.array(list(range(series_set[i].shape[0]))), series_set[j], np.array(list(range(series_set[j].shape[0]))), w=int(series_set.shape[1]))

        return matrix
