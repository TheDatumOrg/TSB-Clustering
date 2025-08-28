import time
import numpy as np
from models.model import BaseClusterModel


class KernelKMeans():
    """
    Kernel K-means implementation based on the Github version
    """
    def __init__(self, n_clusters=3, gamma=0.01, max_iter=100, tol=1e-3, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, K):
        """Fit Kernel K-means with precomputed kernel matrix K"""
        n_samples = K.shape[0]
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Initialize labels randomly
        self.labels_ = np.random.randint(self.n_clusters, size=n_samples)
        
        dist = np.zeros((n_samples, self.n_clusters))
        within_distances = np.zeros(self.n_clusters)
        
        for it in range(self.max_iter):
            dist.fill(0)
            self._compute_dist(K, dist, within_distances, update_within=True)
            
            labels_old = self.labels_.copy()
            self.labels_ = dist.argmin(axis=1)
            
            # Check convergence
            n_same = np.sum((self.labels_ - labels_old) == 0)
            if 1 - float(n_same) / n_samples < self.tol:
                break
        
        return self
    
    def _compute_dist(self, K, dist, within_distances, update_within):
        """Compute distance matrix using kernel trick"""
        for j in range(self.n_clusters):
            mask = self.labels_ == j
            
            if np.sum(mask) == 0:
                dist[:, j] = 0
            else:
                cluster_size = np.sum(mask)
                
                if update_within:
                    # Within-cluster kernel sum
                    K_cluster = K[np.ix_(mask, mask)]
                    within_distances[j] = np.sum(K_cluster) / (cluster_size * cluster_size)
                    dist[:, j] += within_distances[j]
                else:
                    dist[:, j] += within_distances[j]
                
                # Cross-cluster kernel sum
                dist[:, j] -= 2 * np.sum(K[:, mask], axis=1) / cluster_size


class KernelKAvgClusterModel(BaseClusterModel):
    def fit_predict(self, X):
        print(f"Using parameters: {self.params}")
        start_time = time.time()
        
        # Configure parameters
        gamma = self.params.get('gamma', 0.01)
        max_iter = self.params.get('max_iter', 100)
        tol = self.params.get('tol', 1e-3)
        random_state = self.params.get('random_state', None)
        
        if self.distance_matrix is not None:
            # Convert distance matrix to RBF kernel
            if self.distance_name == 'rbf' or 'sink' in str(self.distance_name):
                # Already a similarity/kernel matrix
                K = self.distance_matrix
            else:
                # Convert distance to RBF kernel
                K = np.exp(-gamma * (self.distance_matrix ** 2))
        else:
            # Compute RBF kernel from data
            from sklearn.metrics.pairwise import rbf_kernel
            K = rbf_kernel(X, gamma=gamma)
        
        # Create and fit model
        model = KernelKMeans(
            n_clusters=self.n_clusters,
            gamma=gamma,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state
        )
        
        model.fit(K)
        labels = model.labels_
        
        elapsed = time.time() - start_time
        return labels, elapsed