import time
import numpy as np
from sklearn.mixture import GaussianMixture
from models.model import BaseClusterModel


class GaussianMixtureClusterModel(BaseClusterModel):
    def fit_predict(self, X):
        print(f"Using parameters: {self.params}")
        start_time = time.time()
        
        # Configure clustering parameters
        model_kwargs = {'n_components': self.n_clusters, 'random_state': 42}
        
        # Filter out None values and apply parameters
        valid_params = {k: v for k, v in self.params.items() if v is not None}
        model_kwargs.update(valid_params)
        
        # Gaussian Mixture doesn't support precomputed distance matrices
        if self.distance_matrix is not None:
            print("Warning: Gaussian Mixture does not support precomputed distance matrices. Using original data.")
        
        # Create and fit the model
        model = GaussianMixture(**model_kwargs)
        model.fit(X)
        labels = model.predict(X)
        
        elapsed = time.time() - start_time
        return labels, elapsed