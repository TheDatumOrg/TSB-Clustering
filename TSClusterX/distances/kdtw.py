from tqdm import tqdm
import numpy as np
from distances.distance import DistanceMeasure


class kDTWDistance(DistanceMeasure):
    def kdtw_distance(self, x, y, gamma):
        xlen = len(x)
        ylen = len(y)
        xp = np.zeros(xlen+1)
        yp = np.zeros(ylen+1)
        for i in range(1, xlen+1):
            xp[i] = x[i-1]
        for i in range(1, ylen+1):
            yp[i] = y[i-1]
        xlen = xlen + 1
        ylen = ylen + 1
        x = xp
        y = yp
        length = max(xlen, ylen)
        dp = np.zeros((length, length))
        dp1 = np.zeros((length, length))
        dp2 = np.zeros(length)
        dp2[0] = 1
        for i in range(1, min(xlen, ylen)):
            dp2[i] = self.Dlpr(x[i], y[i], gamma)
        dp[0][0] = 1
        dp1[0][0] = 1
        for i in range(1, xlen):
            dp[i][0] = dp[i - 1][0] * self.Dlpr(x[i], y[1], gamma)
            dp1[i][0] = dp1[i - 1][0] * dp2[i]
        for i in range(1, ylen):
            dp[0][i] = dp[0][i - 1] * self.Dlpr(x[1], y[i], gamma)
            dp1[0][i] = dp1[0][i - 1] * dp2[i]
        for i in range(1, xlen):
            for j in range(1, ylen):
                lcost = self.Dlpr(x[i], y[j], gamma)
                dp[i][j] = (dp[i - 1][j] + dp[i][j - 1] + dp[i - 1][j - 1]) * lcost
                if i ==j:
                    dp1[i][j] = dp1[i - 1][j - 1] * lcost + dp1[i - 1][j] * dp2[i] + dp1[i][j - 1] * dp2[j]
                else:
                    dp1[i][j] = dp1[i - 1][j] * dp2[i] + dp1[i][j - 1] * dp2[j]

        for i in range(0, xlen):
            for j in range(0, ylen):
                dp[i][j] += dp1[i][j]
    
        ans = dp[xlen - 1][ylen - 1]

    def Dlpr(self, x, y, gamma):
        factor=1/3
        minprob=1e-20
        cost = factor*(np.exp(-gamma*np.sum((x - y)**2))+minprob)
        return cost

    def kdtw_norm(self, x, y, gamma):
        sim = self.kdtw_sim(x, y, gamma)/np.sqrt(self.kdtw_sim(x, x, gamma) * self.kdtw_sim(y, y, gamma))
        return sim
    
    def compute(self, series_set):
        n = len(series_set)
        matrix = np.zeros((n, n))
        for i in tqdm(range(n)):
            for j in range(n):
                matrix[i, j] = self.kdtw_distance(series_set[i], series_set[j], gamma=0.125)
        return matrix
