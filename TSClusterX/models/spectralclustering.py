import time
import numpy as np
from sklearn.cluster import SpectralClustering
from models.model import BaseClusterModel


class SpectralClusteringModel(BaseClusterModel):
    def fit_predict(self, X):
        print(f"Using parameters: {self.params}")
        start_time = time.time()
        
        # Configure clustering parameters
        model_kwargs = {'n_clusters': self.n_clusters, 'random_state': 42}
        
        # Filter out None values and apply parameters
        valid_params = {k: v for k, v in self.params.items() if v is not None}
        model_kwargs.update(valid_params)
        
        if self.distance_matrix is not None:
            # Use precomputed affinity matrix (convert distance to affinity)
            # Convert distance matrix to affinity matrix (Gaussian kernel)
            sigma = np.median(self.distance_matrix)
            if sigma == 0:
                sigma = 1.0
            affinity_matrix = np.exp(-self.distance_matrix**2 / (2 * sigma**2))
            np.fill_diagonal(affinity_matrix, 1.0)
            
            model_kwargs['affinity'] = 'precomputed'
            model = SpectralClustering(**model_kwargs)
            labels = model.fit_predict(affinity_matrix)
        else:
            # Use original data with specified or default affinity
            if 'affinity' not in model_kwargs:
                model_kwargs['affinity'] = 'rbf'
            model = SpectralClustering(**model_kwargs)
            labels = model.fit_predict(X)
        
        elapsed = time.time() - start_time
        return labels, elapsed