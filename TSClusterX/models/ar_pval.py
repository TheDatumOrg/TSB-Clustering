import time
import numpy as np
from collections import Counter
from models.model import BaseClusterModel


def find_best_order(ts):
    """Find best AR order using AIC criterion"""
    try:
        from statsmodels.tsa.ar_model import AutoReg
        orders = [1, 2, 3, 4, 5]
        votes = []
        for series in ts:
            min_aic, best_min_order = float('inf'), None
            for order in orders:
                try:
                    res = AutoReg(series, lags=order, old_names=True).fit()
                    if min_aic > res.aic:
                        min_aic = res.aic
                        best_min_order = order
                except:
                    continue
            if best_min_order is not None:
                votes.append(best_min_order)
        if votes:
            votes_counter = Counter(votes)
            return votes_counter.most_common(1)[0][0]
        else:
            return 3  # Default order
    except ImportError:
        print("Warning: statsmodels not available. Using default AR order of 3.")
        return 3


def compute_ar_pvalues(ts, ar_order):
    """Compute AR model p-values"""
    from statsmodels.tsa.ar_model import AutoReg
    coeff_features = np.zeros((ts.shape[0], ar_order + 1))
    
    for i, series in enumerate(ts):
        try:
            res = AutoReg(series, lags=ar_order, old_names=True).fit()
            coeff_features[i, :] = res.pvalues
        except:
            # Use default values if fitting fails
            coeff_features[i, :] = np.ones(ar_order + 1) * 0.5
    
    return coeff_features


class ARPValClusterModel(BaseClusterModel):
    def fit_predict(self, X):
        print(f"Using parameters: {self.params}")
        start_time = time.time()
        
        # Find best AR order
        ar_order = find_best_order(X)
        print(f"Using AR order: {ar_order}")
        
        # Extract AR p-values
        features = compute_ar_pvalues(X, ar_order)
        
        # Compute euclidean distance matrix
        from scipy.spatial.distance import pdist, squareform
        distances = pdist(features, metric='euclidean')
        distance_matrix = squareform(distances)
        
        # Use PAM clustering
        from models.pam import PartitioningAroundMedoids
        labels = PartitioningAroundMedoids(self.n_clusters, np.nan_to_num(distance_matrix))
        
        elapsed = time.time() - start_time
        return labels, elapsed