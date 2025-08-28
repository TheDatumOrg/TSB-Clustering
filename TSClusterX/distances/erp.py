import math
import numpy as np
from distances.distance import DistanceMeasure


class ERPDistance(DistanceMeasure):
    def erp_dist(self, x, y):
        lenx = len(x)
        leny = len(y)

        acc_cost_mat = np.full((lenx, leny), np.inf)

        for i in range(lenx):
            m = 0
            minw = 0
            maxw = leny-1

            for j in range(minw, maxw+1):
                if i + j == 0:
                    acc_cost_mat[i, j] = 0
                elif i == 0:
                    acc_cost_mat[i, j] = acc_cost_mat[i, j-1] + (y[j]-m)**2
                elif j == 0:
                    acc_cost_mat[i, j] = acc_cost_mat[i-1, j] + (x[i]-m)**2
                else:
                    acc_cost_mat[i, j] = min(acc_cost_mat[i-1, j-1] + (x[i] - y[j])**2,
                                         acc_cost_mat[i, j-1] + (y[j] - m)**2,
                                         acc_cost_mat[i-1, j] + (x[i]-m)**2)

        return math.sqrt(acc_cost_mat[lenx-1, leny-1])
    
    def compute(self, series_set):
        n = len(series_set)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                matrix[i, j] = self.erp_dist(series_set[i], series_set[j])
        return matrix
