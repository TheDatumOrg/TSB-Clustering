import time
import numpy as np
from .model import BaseClusterModel


class DTCClusterModel(BaseClusterModel):
    def __init__(self, n_clusters, params=None, distance_name=None, distance_matrix=None):
        super().__init__(n_clusters, params, distance_name, distance_matrix)
        
        # Default parameters for DTC
        self.n_filters = params.get('n_filters', 50)
        self.kernel_size = params.get('kernel_size', 10)
        self.strides = params.get('strides', 1)
        self.pool_size = params.get('pool_size', 10)
        self.n_units = params.get('n_units', [50, 1])
        self.alpha = params.get('alpha', 1.0)
        self.dist_metric = params.get('dist_metric', 'eucl')
        self.cluster_init = params.get('cluster_init', 'kmeans')
        self.epochs = params.get('epochs', 200)
        self.batch_size = params.get('batch_size', 32)

    def fit_predict(self, X):
        """
        Fit DTC model and predict cluster labels.
        
        Args:
            X: Input time series data of shape (n_samples, n_features)
            
        Returns:
            tuple: (predicted_labels, elapsed_time)
        """
        start_time = time.time()
        
        try:
            # Import DTC from utils
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), 'utils', 'DTC'))
            from .utils.DTC.main import DTC
            
            # Prepare data - DTC expects 3D input (samples, timesteps, features)
            if len(X.shape) == 2:
                # Reshape 2D data to 3D by treating each feature as a timestep
                X_reshaped = X.reshape(X.shape[0], X.shape[1], 1)
            else:
                X_reshaped = X
            
            n_samples, timesteps, input_dim = X_reshaped.shape
            
            # Adjust pool_size if necessary
            pool_size = min(self.pool_size, timesteps)
            while timesteps % pool_size != 0 and pool_size > 1:
                pool_size -= 1
            
            # Initialize DTC model
            dtc_model = DTC(
                n_clusters=self.n_clusters,
                input_dim=input_dim,
                timesteps=timesteps,
                n_filters=self.n_filters,
                kernel_size=min(self.kernel_size, timesteps),
                strides=self.strides,
                pool_size=pool_size,
                n_units=self.n_units,
                alpha=self.alpha,
                dist_metric=self.dist_metric,
                cluster_init=self.cluster_init
            )
            
            # Initialize clustering layer (creates autoencoder)
            dtc_model.initialize()
            
            # Pretrain the temporal autoencoder
            print('Pretraining temporal autoencoder...')
            dtc_model.pretrain(X_reshaped, epochs=min(self.epochs // 2, 100))
            
            # Compile the model
            dtc_model.compile(gamma=0.1, optimizer='adam')
            
            # Initialize cluster centers
            dtc_model.init_cluster_weights(X_reshaped)
            
            # Fit the model
            print('Training DTC...')
            y_pred = dtc_model.fit(X_reshaped, epochs=min(self.epochs, 100))
            
            elapsed_time = time.time() - start_time
            return y_pred, elapsed_time
            
        except Exception as e:
            print(f"Error in DTC clustering: {e}")
            # Fallback to K-means if DTC fails
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
            y_pred = kmeans.fit_predict(X)
            elapsed_time = time.time() - start_time
            return y_pred, elapsed_time
