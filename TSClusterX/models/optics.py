import time
import numpy as np
from sklearn.cluster import OPTICS
from models.model import BaseClusterModel


class OPTICSClusterModel(BaseClusterModel):
    def fit_predict(self, X):
        print(f"Using parameters: {self.params}")
        start_time = time.time()
        
        # Configure clustering parameters
        model_kwargs = {}
        
        # Filter out None values and apply parameters
        valid_params = {k: v for k, v in self.params.items() if v is not None}
        model_kwargs.update(valid_params)
        
        if self.distance_matrix is not None:
            # Use precomputed distance matrix
            model_kwargs['metric'] = 'precomputed'
            model = OPTICS(**model_kwargs)
            labels = model.fit_predict(self.distance_matrix)
        else:
            # Use original data with specified or default metric
            if 'metric' not in model_kwargs:
                model_kwargs['metric'] = 'euclidean'
            model = OPTICS(**model_kwargs)
            labels = model.fit_predict(X)
        
        elapsed = time.time() - start_time
        return labels, elapsed