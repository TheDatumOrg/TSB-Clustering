import time
import numpy as np
from models.model import BaseClusterModel


def compute_exponential_smoothing_features(ts):
    """Compute exponential smoothing coefficients"""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    coeff_features = np.zeros((ts.shape[0], 3))
    
    for i, series in enumerate(ts):
        try:
            # Fit exponential smoothing model
            fit1 = ExponentialSmoothing(
                series, 
                seasonal_periods=4, 
                trend='add',
                seasonal='add', 
                use_boxcox=False, 
                initialization_method="heuristic"
            ).fit()
            
            # Extract smoothing parameters
            params = [
                fit1.params.get('smoothing_level', 0.0),
                fit1.params.get('smoothing_trend', 0.0),
                fit1.params.get('damping_trend', 0.0)
            ]
            coeff_features[i, :] = np.nan_to_num(np.array(params))
            
        except Exception as e:
            print(f"Warning: Failed to fit exponential smoothing for series {i}: {e}")
            # Use default values
            coeff_features[i, :] = [0.3, 0.1, 0.8]  # Default smoothing parameters
    
    return coeff_features


class ESCoeffClusterModel(BaseClusterModel):
    def fit_predict(self, X):
        print(f"Using parameters: {self.params}")
        start_time = time.time()
        
        # Extract exponential smoothing features
        features = compute_exponential_smoothing_features(X)
        
        # Compute euclidean distance matrix
        from scipy.spatial.distance import pdist, squareform
        distances = pdist(features, metric='euclidean')
        distance_matrix = squareform(distances)
        
        # Use PAM clustering
        from models.pam import PartitioningAroundMedoids
        labels = PartitioningAroundMedoids(self.n_clusters, distance_matrix)
        
        elapsed = time.time() - start_time
        return labels, elapsed