import time
import numpy as np
from sklearn.cluster import Birch
from models.model import BaseClusterModel


class BirchClusterModel(BaseClusterModel):
    def fit_predict(self, X):
        print(f"Using parameters: {self.params}")
        start_time = time.time()
        
        # Configure clustering parameters
        model_kwargs = {'n_clusters': self.n_clusters}
        
        # Filter out None values and apply parameters
        valid_params = {k: v for k, v in self.params.items() if v is not None}
        model_kwargs.update(valid_params)
        
        # BIRCH doesn't support precomputed distance matrices
        if self.distance_matrix is not None:
            print("Warning: BIRCH does not support precomputed distance matrices. Using original data.")
        
        # Create and fit the model
        model = Birch(**model_kwargs)
        labels = model.fit_predict(X)
        
        elapsed = time.time() - start_time
        return labels, elapsed