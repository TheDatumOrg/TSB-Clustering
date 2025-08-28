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


def lpcc_transform(coef):
    """Convert AR coefficients to LPCC coefficients"""
    ar_order = len(coef) - 1
    lpcc_coeffs = [-coef[0]]
    for n in range(1, ar_order + 1):
        upbound = (ar_order + 1 if n > ar_order else n)
        lpcc_coef = -np.sum(i * lpcc_coeffs[i] * coef[n - i - 1]
                            for i in range(1, upbound)) * 1. / upbound
        lpcc_coef -= coef[n - 1] if n <= len(coef) else 0
        lpcc_coeffs.append(lpcc_coef)
    return np.array(lpcc_coeffs)


def compute_ar_features(ts, ar_order, transform_type='lpcc'):
    """Compute AR-based features"""
    from statsmodels.tsa.ar_model import AutoReg
    coeff_features = np.zeros((ts.shape[0], ar_order + 1))
    
    for i, series in enumerate(ts):
        try:
            res = AutoReg(series, lags=ar_order, old_names=True).fit()
            if transform_type == 'lpcc':
                coeff_features[i, :] = lpcc_transform(res.params)
            elif transform_type == 'coeff':
                coeff_features[i, :] = res.params
            elif transform_type == 'pval':
                coeff_features[i, :] = res.pvalues
        except:
            # Use zeros if fitting fails
            coeff_features[i, :] = np.zeros(ar_order + 1)
    
    return coeff_features


class LPCCClusterModel(BaseClusterModel):
    def fit_predict(self, X):
        print(f"Using parameters: {self.params}")
        start_time = time.time()
        
        # Configure parameters
        ar_coeff_transforms = self.params.get('ar_coeff_transforms', 'lpcc')
        
        # Find best AR order
        ar_order = find_best_order(X)
        print(f"Using AR order: {ar_order}")
        
        # Extract AR coefficients and transform to LPCC
        features = compute_ar_features(X, ar_order, ar_coeff_transforms)
        
        # Compute euclidean distance matrix
        from scipy.spatial.distance import pdist, squareform
        distances = pdist(features, metric='euclidean')
        distance_matrix = squareform(distances)
        
        # Use PAM clustering
        from models.pam import PartitioningAroundMedoids
        labels = PartitioningAroundMedoids(self.n_clusters, np.nan_to_num(distance_matrix))
        
        elapsed = time.time() - start_time
        return labels, elapsed