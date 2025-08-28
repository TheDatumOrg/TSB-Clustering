import numpy as np
from distances.distance import DistanceMeasure


class LCSSDistance(DistanceMeasure):
    def lcss_dist(self, x, y, w):
        lenx = len(x)
        leny = len(y)
        epsilon = 0.2
        if w == None:
            w = max(lenx, leny)
        D = np.zeros((lenx, leny))
        for i in range(lenx):
            wmin = max(0, i-w)
            wmax = min(leny-2, i+w)
            for j in range(wmin, wmax+1):
                if i + j == 0:
                    if abs(x[i]-y[j]) <= epsilon:
                        D[i][j] = 1
                    else:
                        D[i][j] = 0
                elif i == 0:
                    if abs(x[i]-y[j]) <= epsilon:
                        D[i][j] = 1
                    else:
                        D[i][j] =  D[i][j-1]
                elif j ==0:
                    if abs(x[i]-y[j]) <= epsilon:
                        D[i][j] = 1
                    else:
                        D[i][j] =  D[i-1][j]
                else:
                    if abs(x[i]-y[j]) <= epsilon:
                        D[i][j] = max(D[i-1][j-1]+1,
                                      D[i-1][j],
                                      D[i][j+1])
                    else:
                        D[i][j] = max(D[i-1][j-1],
                                      D[i-1][j],
                                      D[i][j+1])
        result = D[lenx-1, leny -1]
        return 1 - result/min(len(x),len(y))
    
    def compute(self, series_set):
        n = len(series_set)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                matrix[i, j] = self.lcss_dist(series_set[i], series_set[j], w=int(len(series_set[i])/20))
        return matrix
