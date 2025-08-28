import time
import numpy as np
from models.model import BaseClusterModel


class KShapeClusterModel(BaseClusterModel):
    def fit_predict(self, X):
        print(f"Using parameters: {self.params}")
        start_time = time.time()
        
        # Import kshape
        from kshape.core import kshape
        
        # Prepare data - kshape expects 3D array (n_samples, n_timepoints, 1)
        ts = np.expand_dims(X, axis=2)
        
        # Get parameters with defaults
        centroid_init = self.params.get('centroid_init', 'zero')
        max_iter = self.params.get('max_iter', 100)
        
        # Apply k-shape clustering
        kshape_model = kshape(ts, self.n_clusters, centroid_init=centroid_init, max_iter=max_iter)
        
        # Extract predictions from k-shape result
        # kshape returns a list of (centroid, indices) tuples
        predictions = np.zeros(ts.shape[0])
        for i in range(self.n_clusters):
            centroid, indices = kshape_model[i]
            predictions[indices] = i
        
        elapsed = time.time() - start_time
        return predictions.astype(int), elapsed
