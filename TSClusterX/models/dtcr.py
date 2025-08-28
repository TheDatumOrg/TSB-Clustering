import time
import numpy as np
from .model import BaseClusterModel


class DTCRClusterModel(BaseClusterModel):
    def __init__(self, n_clusters, params=None, distance_name=None, distance_matrix=None):
        super().__init__(n_clusters, params, distance_name, distance_matrix)
        
        # Default parameters for DTCR
        self.encoder_hidden_units = params.get('encoder_hidden_units', [100, 50, 50, 30])
        self.lambda_param = params.get('lambda', 1e-4)
        self.dilations = params.get('dilations', [1, 2, 4, 8])
        self.epochs = params.get('epochs', 100)
        self.training_samples_num = params.get('training_samples_num', 100)

    def fit_predict(self, X):
        """
        Fit DTCR model and predict cluster labels.
        
        Args:
            X: Input time series data of shape (n_samples, n_features)
            
        Returns:
            tuple: (predicted_labels, elapsed_time)
        """
        start_time = time.time()
        
        try:
            # Import DTCR from utils
            import sys
            import os
            import tensorflow.compat.v1 as tf
            tf.disable_v2_behavior()
            
            # Add the parent directory to path so we can import the DTCR package
            utils_path = os.path.join(os.path.dirname(__file__), 'utils')
            if utils_path not in sys.path:
                sys.path.insert(0, utils_path)
            
            from DTCR.main import dtcr_clustering
            
            # Create fake labels for DTCR (it expects labels for pretraining)
            fake_labels = np.arange(X.shape[0]) % self.n_clusters
            
            # Parameters for DTCR
            params = [self.encoder_hidden_units, self.lambda_param, self.dilations]
            
            # Run DTCR clustering
            print('Training DTCR...')
            best_inertia, y_pred = dtcr_clustering(
                X, fake_labels, self.n_clusters, params, best=True
            )
            
            elapsed_time = time.time() - start_time
            return y_pred, elapsed_time
            
        except Exception as e:
            import traceback
            print(f"Error in DTCR clustering: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            # Fallback to K-means if DTCR fails
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
            y_pred = kmeans.fit_predict(X)
            elapsed_time = time.time() - start_time
            return y_pred, elapsed_time
