import time
import numpy as np
from models.model import BaseClusterModel


def catch22_features(ts):
    """
    Extract Catch22 features from time series using the actual catch22 library
    """
    import pycatch22
    features = np.zeros((ts.shape[0], 22))
    for i in range(ts.shape[0]):
        catchOut = pycatch22.catch22_all(ts[i])
        features[i, :] = catchOut['values']
    return features

class Catch22ClusterModel(BaseClusterModel):
    def fit_predict(self, X):
        print(f"Using parameters: {self.params}")
        start_time = time.time()
        
        # Check distance measure requirement (from original implementation)
        if self.distance_name and self.distance_name != 'euclidean':
            raise ValueError("Catch22 only supports euclidean distance")
        
        if self.distance_matrix is not None and self.params.get('precomputed', False):
            # Use precomputed catch22 features distance matrix
            dist_mat = np.nan_to_num(self.distance_matrix)
            dist_mat[dist_mat < 0] = 0
        else:
            # Extract catch22 features and compute distance matrix
            features = np.nan_to_num(catch22_features(X))
            
            # Compute euclidean distance matrix (simulate dm.euclidean behavior)
            from scipy.spatial.distance import pdist, squareform
            distances = pdist(features, metric='euclidean')
            dist_mat = squareform(distances)
        
        # Use PAM clustering (as in original implementation)
        from models.pam import PartitioningAroundMedoids
        labels = PartitioningAroundMedoids(self.n_clusters, dist_mat)
        
        elapsed = time.time() - start_time
        return labels, elapsed