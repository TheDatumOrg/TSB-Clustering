import time
import numpy as np
from aeon.clustering import KASBA
from models.model import BaseClusterModel


class KASBAClusterModel(BaseClusterModel):
    def fit_predict(self, X):
        print(f"Using parameters: {self.params}")
        start_time = time.time()
        
        model_kwargs = {'n_clusters': self.n_clusters}
        
        default_params = {
            'max_iter': 100,
            'distance': 'msm'
        }
        
        model_kwargs.update(default_params)
        
        valid_params = {k: v for k, v in self.params.items() if v is not None}
        model_kwargs.update(valid_params)

        model = KASBA(**model_kwargs)
        
        model.fit(X)
        labels = model.predict(X)
        
        elapsed = time.time() - start_time
        return labels, elapsed
