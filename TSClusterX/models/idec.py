import time
import numpy as np
from .model import BaseClusterModel


class IDECClusterModel(BaseClusterModel):
    def __init__(self, n_clusters, params=None, distance_name=None, distance_matrix=None):
        super().__init__(n_clusters, params, distance_name, distance_matrix)
        
        # Default parameters for IDEC
        self.dims = params.get('dims', [784, 500, 500, 2000, 10])  # AutoEncoder architecture
        self.alpha = params.get('alpha', 1.0)  # IDEC loss parameter
        self.batch_size = params.get('batch_size', 256)
        self.maxiter = params.get('maxiter', 2e4)
        self.update_interval = params.get('update_interval', 140)
        self.tol = params.get('tol', 1e-3)
        self.ae_weights = params.get('ae_weights', None)
        self.save_dir = params.get('save_dir', './results/temp')
        self.gamma = params.get('gamma', 0.1)  # IDEC specific parameter

    def fit_predict(self, X):
        """
        Fit IDEC model and predict cluster labels.
        
        Args:
            X: Input time series data of shape (n_samples, n_features)
            
        Returns:
            tuple: (predicted_labels, elapsed_time)
        """
        start_time = time.time()
        
        try:
            # Import IDEC from utils
            import sys
            import os
            idec_path = os.path.join(os.path.dirname(__file__), 'utils', 'IDEC')
            sys.path.append(idec_path)
            
            # Import IDEC class
            from IDEC import IDEC
            
            # Adjust dimensions based on input data
            if X.shape[1] != self.dims[0]:
                self.dims[0] = X.shape[1]
            
            # Initialize IDEC model
            idec_model = IDEC(dims=self.dims, n_clusters=self.n_clusters, alpha=self.alpha)
            
            # Pretrain autoencoder if no weights are provided
            if self.ae_weights is None:
                print('Pretraining autoencoder...')
                idec_model.pretrain(X, batch_size=self.batch_size, epochs=300, optimizer='adam')
            else:
                idec_model.load_weights(self.ae_weights)
            
            # Compile the model for clustering
            idec_model.compile(loss=['kld', 'mse'], loss_weights=[self.gamma, 1], optimizer='adam')
            
            # Cluster
            fit_result = idec_model.fit(X, 
                                      batch_size=self.batch_size,
                                      maxiter=int(self.maxiter),
                                      update_interval=self.update_interval,
                                      tol=self.tol,
                                      save_dir=self.save_dir)
            
            # Extract predicted labels from fit result
            # fit returns: (best_loss, best_y_pred, best_inertia, loss, y_pred, inertia)
            y_pred = fit_result[1]  # best_y_pred
            
            elapsed_time = time.time() - start_time
            return y_pred, elapsed_time
            
        except Exception as e:
            print(f"Error in IDEC clustering: {e}")
            # Fallback to K-means if IDEC fails
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
            y_pred = kmeans.fit_predict(X)
            elapsed_time = time.time() - start_time
            return y_pred, elapsed_time
