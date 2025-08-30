import time
import torch
import numpy as np
from sklearn.cluster import KMeans
from models.model import BaseClusterModel

try:
    from chronos import ChronosPipeline
except ImportError:
    ChronosPipeline = None


class ChronosClusterModel(BaseClusterModel):
    def fit_predict(self, X):
        if ChronosPipeline is None:
            raise ImportError("ChronosPipeline not available. Please install the chronos package.")
        
        print(f"Using parameters: {self.params}")
        start_time = time.time()
        
        # Default parameters
        model_name = self.params.get('model_name', './chronos-t5-small')
        device_map = self.params.get('device_map', 'cuda')
        torch_dtype_str = self.params.get('torch_dtype', 'torch.bfloat16')
        
        # Convert string to actual torch dtype
        if torch_dtype_str == 'torch.bfloat16':
            torch_dtype = torch.bfloat16
        elif torch_dtype_str == 'torch.float16':
            torch_dtype = torch.float16
        elif torch_dtype_str == 'torch.float32':
            torch_dtype = torch.float32
        else:
            torch_dtype = torch.bfloat16
        
        # KMeans parameters
        kmeans_init = self.params.get('kmeans_init', 'random')
        kmeans_n_init = self.params.get('kmeans_n_init', 1)
        kmeans_max_iter = self.params.get('kmeans_max_iter', 100)
        kmeans_tol = self.params.get('kmeans_tol', 1e-4)
        
        # Initialize pipeline
        pipeline = ChronosPipeline.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
        )
        
        # Extract embeddings
        rep = []
        for t in X:
            context = torch.tensor(t)
            embeddings, tokenizer_state = pipeline.embed(context)
            representation = torch.mean(embeddings, dim=1).to(torch.float32)
            rep.append(representation.squeeze(dim=0).cpu().detach().numpy())
            
        rep = np.array(rep)
        
        # Perform clustering
        kmeans = KMeans(
            n_clusters=self.n_clusters, 
            init=kmeans_init, 
            n_init=kmeans_n_init,
            max_iter=kmeans_max_iter,
            tol=kmeans_tol,
            random_state=42
        )
        labels = kmeans.fit_predict(rep)

        elapsed = time.time() - start_time
        return labels, elapsed
