import time
import numpy as np
from .model import BaseClusterModel


class DCNClusterModel(BaseClusterModel):
    def __init__(self, n_clusters, params=None, distance_name=None, distance_matrix=None):
        super().__init__(n_clusters, params, distance_name, distance_matrix)
        
        # Default parameters for DCN
        self.dims = params.get('dims', [784, 500, 500, 2000, 10])  # AutoEncoder architecture
        self.init = params.get('init', 'glorot_uniform')
        self.act = params.get('act', 'relu')
        self.lr = params.get('lr', 0.001)
        self.epochs = params.get('epochs', 300)
        self.batch_size = params.get('batch_size', 256)
        self.tol = params.get('tol', 1e-3)
        self.update_interval = params.get('update_interval', 140)
        self.save_dir = params.get('save_dir', './results/temp')

    def fit_predict(self, X):
        """
        Fit DCN model and predict cluster labels.
        
        Args:
            X: Input time series data of shape (n_samples, n_features)
            
        Returns:
            tuple: (predicted_labels, elapsed_time)
        """
        start_time = time.time()

        # Import DCN from utils
        import sys
        import os
        import numpy as np
        from .utils.DCN_keras.DCN import dcn_clustering
        
        # Prepare parameters for dcn_clustering function
        # The function expects: dcn_clustering(ts, labels, nclusters, params, best=True)
        # where params[0] should be the architecture
        
        # Create architecture (hidden layers excluding input and output)
        architecture = np.array(self.dims[1:])  # Skip input dimension
        params = [architecture]
        
        # Call the dcn_clustering function
        print('Running DCN clustering...')
        _, y_pred = dcn_clustering(X, np.zeros(len(X)), self.n_clusters, params, best=True)
        
        elapsed_time = time.time() - start_time
        return y_pred, elapsed_time
        
