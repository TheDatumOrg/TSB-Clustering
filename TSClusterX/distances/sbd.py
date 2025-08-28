import numpy as np
from distances.distance import DistanceMeasure


class SBDDistance(DistanceMeasure):
    def calculate_SBD_distance(self, x, y):
        cc = self.NCCc_pairwise(x, y)
        maxidx = np.argmax(cc)
        dist = 1 - cc[maxidx]
        return dist

    def _next_pow_2(self, x):
        return 1<<(x-1).bit_length()

    def NCCc_pairwise(self, x, y):
        l = self._next_pow_2(2 * len(x) - 1)
        cc = np.fft.ifft(np.fft.fft(x, n=l) * np.conjugate(np.fft.fft(y, n=l)))
        return cc / (np.linalg.norm(x) * np.linalg.norm(y))
    
    def compute(self, series_set):
        n = len(series_set)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                matrix[i, j] = self.calculate_SBD_distance(series_set[i], series_set[j])
        return matrix
