import time
from sklearn.cluster import AgglomerativeClustering
from models.model import BaseClusterModel


class AgglomerativeClusterModel(BaseClusterModel):
    def fit_predict(self, X):
        print(f"Using parameters: {self.params}")
        start_time = time.time()
        
        # Configure clustering based on available distance information
        if self.distance_matrix is not None:
            # Use precomputed distance matrix
            # Ward linkage cannot be used with precomputed distances
            linkage = self.params.get('linkage', 'complete')
            if linkage == 'ward':
                print("Warning: Ward linkage cannot be used with precomputed distances. Using 'complete' linkage instead.")
                linkage = 'complete'
            
            # Filter out parameters that are incompatible with precomputed distances
            valid_params = {k: v for k, v in self.params.items() 
                          if k not in ['linkage', 'metric'] and v is not None}
            
            model = AgglomerativeClustering(
                n_clusters=self.n_clusters, 
                metric='precomputed', 
                linkage=linkage,
                **valid_params
            )
            labels = model.fit_predict(self.distance_matrix)
        elif self.distance_name == 'euclidean' or self.distance_name is None:
            # Use euclidean distance (default sklearn behavior)
            # Apply parameters from the loaded config
            model_kwargs = {'n_clusters': self.n_clusters}
            # Filter out None values
            valid_params = {k: v for k, v in self.params.items() if v is not None}
            model_kwargs.update(valid_params)
            model = AgglomerativeClustering(**model_kwargs)
            labels = model.fit_predict(X)
        else:
            # For other distance types, fall back to euclidean if no precomputed matrix
            print(f"Warning: Distance '{self.distance_name}' specified but no distance matrix provided. Using euclidean distance.")
            model_kwargs = {'n_clusters': self.n_clusters}
            # Filter out None values
            valid_params = {k: v for k, v in self.params.items() if v is not None}
            model_kwargs.update(valid_params)
            model = AgglomerativeClustering(**model_kwargs)
            labels = model.fit_predict(X)
            
        elapsed = time.time() - start_time
        return labels, elapsed
        