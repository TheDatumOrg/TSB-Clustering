import time
import numpy as np
from models.model import BaseClusterModel


class KSCClusterModel(BaseClusterModel):
    def fit_predict(self, X):
        print(f"Using parameters: {self.params}")
        start_time = time.time()
        
        # KSC doesn't support precomputed distance matrices
        if self.distance_matrix is not None:
            print("Warning: KSC does not support precomputed distance matrices. Using original data.")
        
        # Extract parameters with defaults
        max_iter = self.params.get('max_iter', 100)
        n_runs = self.params.get('n_runs', 10)  # Default n_runs from original implementation
        random_state = self.params.get('random_state', None)
        
        # Set random seed if provided
        if random_state is not None:
            np.random.seed(random_state)
        

        # Try to import the actual pyksc implementation from utils
        from models.utils.pyksc import ksc
        
        # Add small constant to avoid numerical issues (as in original)
        X_adjusted = X + 0.0001
        
        # Use the actual ksc implementation with parameters from config
        print(f"Running KSC with max_iter={max_iter}, n_runs={n_runs}, n_clusters={self.n_clusters}")
        _, predictions, _, _ = ksc.ksc(X_adjusted, self.n_clusters, max_iter, n_runs)
        
        
        elapsed = time.time() - start_time
        return predictions, elapsed