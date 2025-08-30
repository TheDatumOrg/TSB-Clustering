import time
import numpy as np
from sklearn.cluster import SpectralClustering
from models.model import BaseClusterModel


class SpectralClusteringModel(BaseClusterModel):
    def fit_predict(self, X):
        print(f"Using parameters: {self.params}")
        start_time = time.time()
        
        # Configure clustering parameters
        model_kwargs = {'n_clusters': self.n_clusters}
        
        # Filter out None values and apply parameters
        valid_params = {k: v for k, v in self.params.items() if v is not None}
        model_kwargs.update(valid_params)
        
        if self.distance_matrix is not None:
            affinity_matrix = self.distance_matrix
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