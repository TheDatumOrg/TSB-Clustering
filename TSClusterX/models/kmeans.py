import time
import numpy as np
from sklearn.cluster import KMeans
from models.model import BaseClusterModel


class KMeansClusterModel(BaseClusterModel):
    def fit_predict(self, X):
        print(f"Using parameters: {self.params}")
        start_time = time.time()
        
        # Configure clustering
        model_kwargs = {'n_clusters': self.n_clusters}
        
        # Filter out None values and apply parameters
        valid_params = {k: v for k, v in self.params.items() if v is not None}
        model_kwargs.update(valid_params)
        
        # Create and fit the model
        model = KMeans(**model_kwargs)
        
        if self.distance_matrix is not None:
            # For precomputed distances, we need to use the original data
            print("Warning: K-means does not support precomputed distance matrices. Using original data.")
        
        labels = model.fit_predict(X)
        
        elapsed = time.time() - start_time
        return labels, elapsed
