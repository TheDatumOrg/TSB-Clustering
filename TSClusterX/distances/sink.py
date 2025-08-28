import numpy as np
from distances.distance import DistanceMeasure


class SINKDistance(DistanceMeasure):
    def _next_pow_2(self, x):
        return 1<<(x-1).bit_length()

    def NCCc_pairwise(self, x, y):
        l = self._next_pow_2(2 * len(x) - 1)
        cc = np.fft.ifft(np.fft.fft(x, n=l) * np.conjugate(np.fft.fft(y, n=l)))
        return cc / (np.linalg.norm(x) * np.linalg.norm(y))

    def sum_exp_NCCc(self, x, y, gamma):
        sim = sum(np.exp(gamma*self.NCCc_pairwise(x,y)))
        return sim

    def calculate_sink_distance(self, x, y, sum_exp_NCCc_xx, sum_exp_NCCc_yy, gamma):
        sim = self.sum_exp_NCCc(x, y, gamma) / (np.sqrt(sum_exp_NCCc_xx * sum_exp_NCCc_yy))
        return sim
    
    def store_sum_exp_NCCc(self, ts, gamma):
        sum_exp_NCCc_list = []
        for i in range(ts.shape[0]):
            sum_exp_NCCc_list.append(self.sum_exp_NCCc(ts[i], ts[i], gamma))
        return sum_exp_NCCc_list

    def compute(self, series_set):
        n = len(series_set)
        matrix = np.zeros((n, n))
        gamma = 5

        sum_exp_NCCc_list = self.store_sum_exp_NCCc(series_set, gamma)
        for i in range(n):
            for j in range(n):
                matrix[i, j] = self.calculate_sink_distance(series_set[i], series_set[j], sum_exp_NCCc_list[i], sum_exp_NCCc_list[j], gamma)
        return matrix
