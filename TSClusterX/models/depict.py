import time
import numpy as np
from .model import BaseClusterModel


class DEPICTClusterModel(BaseClusterModel):
    def __init__(self, n_clusters, params=None, distance_name=None, distance_matrix=None):
        super().__init__(n_clusters, params, distance_name, distance_matrix)
        
        # Default parameters for DEPICT
        self.epochs = params.get('epochs', 100)
        self.batch_size = params.get('batch_size', 32)
        self.learning_rate = params.get('learning_rate', 0.001)
        self.device = params.get('device', 'cpu')
        self.alpha = params.get('alpha', 1.0)

    def fit_predict(self, X):
        """
        Fit DEPICT model and predict cluster labels.
        
        Args:
            X: Input time series data of shape (n_samples, n_features)
            
        Returns:
            tuple: (predicted_labels, elapsed_time)
        """
        start_time = time.time()

        # Import DEPICT from utils
        import sys
        import os
        import torch
        utils_path = os.path.join(os.path.dirname(__file__), 'utils')
        if utils_path not in sys.path:
            sys.path.insert(0, utils_path)
        import DEPICT
        
        # Create dummy labels for the depict function (it expects labels but doesn't use them for clustering)
        dummy_labels = np.zeros(X.shape[0])
        
        # Call the DEPICT clustering function
        y_pred = DEPICT.depict(X, dummy_labels, self.n_clusters)
        
        elapsed_time = time.time() - start_time
        return y_pred, elapsed_time
        
