import time
import numpy as np
from .model import BaseClusterModel


class DECClusterModel(BaseClusterModel):
    def __init__(self, n_clusters, params=None, distance_name=None, distance_matrix=None):
        super().__init__(n_clusters, params, distance_name, distance_matrix)
        
        # Default parameters for DEC
        self.dims = params.get('dims', [784, 500, 500, 2000, 10])  # AutoEncoder architecture
        self.alpha = params.get('alpha', 1.0)  # DEC loss parameter
        self.batch_size = params.get('batch_size', 256)
        self.maxiter = params.get('maxiter', 2e4)
        self.update_interval = params.get('update_interval', 140)
        self.tol = params.get('tol', 1e-3)
        self.ae_weights = params.get('ae_weights', None)
        self.save_dir = params.get('save_dir', './results/temp')

    def fit_predict(self, X):
        """
        Fit DEC model and predict cluster labels.
        
        Args:
            X: Input time series data of shape (n_samples, n_features)
            
        Returns:
            tuple: (predicted_labels, elapsed_time)
        """
        start_time = time.time()
        
        # Import DEC from utils
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), 'utils', 'IDEC'))
        from DEC import DEC
        
        # Adjust dimensions based on input data
        if X.shape[1] != self.dims[0]:
            self.dims[0] = X.shape[1]
        
        # Initialize DEC model
        dec_model = DEC(dims=self.dims, n_clusters=self.n_clusters, alpha=self.alpha)
        
        # Pretrain autoencoder if no weights are provided
        if self.ae_weights is None:
            print('Pretraining autoencoder...')
            dec_model.pretrain(X, batch_size=self.batch_size, epochs=300, optimizer='adam')
        else:
            dec_model.autoencoder.load_weights(self.ae_weights)
        
        # Compile the DEC model
        dec_model.compile(loss='kld', optimizer='adam')
        
        # Cluster
        fit_result = dec_model.fit(X, 
                                    batch_size=self.batch_size,
                                    maxiter=int(self.maxiter),
                                    update_interval=self.update_interval,
                                    tol=self.tol,
                                    save_dir=self.save_dir)
        
        # Extract predicted labels from the tuple result
        # fit returns: (best_loss, best_y_pred, best_inertia, loss, y_pred, inertia)
        y_pred = fit_result[1]  # best_y_pred
        
        elapsed_time = time.time() - start_time
        return y_pred, elapsed_time
        
