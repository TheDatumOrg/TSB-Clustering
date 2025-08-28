import time
import numpy as np
from .model import BaseClusterModel


class UShapeletsClusterModel(BaseClusterModel):
    def __init__(self, n_clusters, params=None, distance_name=None, distance_matrix=None):
        super().__init__(n_clusters, params, distance_name, distance_matrix)
        
    def fit_predict(self, X):
        # Import the UShapelet function from our utils
        from .utils.UShapeletPythonCode.RunManyClusters_Fast import UShapelet
        
        cluster_time_start = time.time()
        
        # Create dummy labels for clustering (as required by the algorithm)
        # Since this is unsupervised learning, we create dummy labels
        dummy_labels = np.arange(len(X))
        
        # Call the UShapelet function
        predictions = UShapelet(X, dummy_labels.copy())
        
        cluster_timing = time.time() - cluster_time_start
        
        # Convert to integer labels
        predictions = predictions.astype(int)
        
        return predictions, cluster_timing