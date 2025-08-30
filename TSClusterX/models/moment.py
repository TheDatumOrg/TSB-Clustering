import time
import torch
import numpy as np
from sklearn.cluster import KMeans
from models.model import BaseClusterModel

try:
    from momentfm import MOMENTPipeline
except ImportError:
    MOMENTPipeline = None


class MomentClusterModel(BaseClusterModel):
    def fit_predict(self, X):
        if MOMENTPipeline is None:
            raise ImportError("MOMENTPipeline not available. Please install the momentfm package.")
        
        print(f"Using parameters: {self.params}")
        start_time = time.time()
        
        # Default parameters
        model_name = self.params.get('model_name', 'AutonLab/MOMENT-1-large')
        task_name = self.params.get('task_name', 'embedding')
        
        # KMeans parameters
        kmeans_init = self.params.get('kmeans_init', 'random')
        kmeans_n_init = self.params.get('kmeans_n_init', 1)
        kmeans_max_iter = self.params.get('kmeans_max_iter', 100)
        kmeans_tol = self.params.get('kmeans_tol', 1e-4)
        
        # Initialize model
        model = MOMENTPipeline.from_pretrained(
            model_name, 
            model_kwargs={"task_name": task_name},
        )
        model.init()
        
        # Check if CUDA is available
        if torch.cuda.is_available():
            model.cuda()
            device = 'cuda'
        else:
            device = 'cpu'
        
        # Extract embeddings
        rep = []
        for t in X:
            context = torch.tensor(t).unsqueeze(0).unsqueeze(0).to(torch.float32)
            if device == 'cuda':
                context = context.cuda()
            
            embeddings = model(context)
            representation = embeddings.embeddings.squeeze(0).detach().cpu().numpy()
            rep.append(representation)
            
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
