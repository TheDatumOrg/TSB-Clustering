import time
import math
import numpy as np
from models.model import BaseClusterModel


class DensityPeakCluster(object):
    """
    Density Peak Cluster implementation adapted from Github version.
    """
    def __init__(self,
                 dc=None,
                 silence=True,
                 gauss_cutoff=True,
                 threshold_metric='kneepoint',
                 density_threshold=None,
                 distance_threshold=None,
                 anormal=True):
        self.dc = dc
        self.silence = silence
        self.gauss_cutoff = gauss_cutoff
        self.threshold_metric = threshold_metric
        self.density_threshold = density_threshold
        self.distance_threshold = distance_threshold
        self.anormal = anormal
        
    def fit(self, distance_matrix, n_clusters):
        """
        Fit the density peak clustering algorithm.
        
        Args:
            distance_matrix: Pre-computed distance matrix
            n_clusters: Number of clusters
            
        Returns:
            labels: Cluster labels
        """
        self.n_id = distance_matrix.shape[0]
        self.distance = distance_matrix
        
        # Calculate dc if not provided
        if self.dc is None:
            self.dc = self._select_dc()
            
        # Calculate density (rho) and minimum distance to higher density points (delta)
        self._local_density()
        self._min_distance()
        
        # Find cluster centers
        centers = self._find_centers_auto(n_clusters)
        
        # Assign labels based on centers
        labels = self._cluster_PD(centers)
        
        return labels
        
    def _select_dc(self):
        """Select the cutoff distance dc automatically."""
        # Use 2% of max distance as default
        max_dis = np.max(self.distance)
        return max_dis * 0.02
        
    def _local_density(self):
        """Calculate the local density for each point."""
        self.rho = np.zeros(self.n_id)
        
        for i in range(self.n_id):
            if self.gauss_cutoff:
                # Gaussian cutoff
                self.rho[i] = np.sum(np.exp(-((self.distance[i] / self.dc) ** 2))) - 1
            else:
                # Hard cutoff
                self.rho[i] = np.sum(self.distance[i] < self.dc) - 1
                
    def _min_distance(self):
        """Calculate minimum distance to higher density points."""
        self.delta = np.zeros(self.n_id)
        self.nneigh = np.zeros(self.n_id, dtype=int)
        
        # Sort by density (descending)
        rho_sorted = np.argsort(-self.rho)
        
        for i, idx in enumerate(rho_sorted):
            if i == 0:
                # Point with highest density
                self.delta[idx] = np.max(self.distance[idx])
            else:
                # Find minimum distance to higher density points
                higher_density_points = rho_sorted[:i]
                distances_to_higher = self.distance[idx][higher_density_points]
                min_idx = np.argmin(distances_to_higher)
                self.delta[idx] = distances_to_higher[min_idx]
                self.nneigh[idx] = higher_density_points[min_idx]
                
    def _find_centers_auto(self, n_clusters):
        """Automatically find cluster centers."""
        # Calculate gamma (rho * delta)
        gamma = self.rho * self.delta
        
        # Select top n_clusters points with highest gamma as centers
        centers = np.argsort(-gamma)[:n_clusters]
        
        return centers
        
    def _cluster_PD(self, centers):
        """Assign cluster labels based on centers."""
        labels = np.full(self.n_id, -1)
        
        # Assign center labels
        for i, center in enumerate(centers):
            labels[center] = i
            
        # Sort by density (descending)
        rho_sorted = np.argsort(-self.rho)
        
        # Assign other points
        for idx in rho_sorted:
            if labels[idx] == -1:  # Not a center
                labels[idx] = labels[self.nneigh[idx]]
                
        return labels


class DensityPeaksClusterModel(BaseClusterModel):
    def fit_predict(self, X):
        print(f"Using parameters: {self.params}")
        start_time = time.time()
        
        # Configure clustering parameters
        cluster_params = {
            'dc': self.params.get('dc', None),
            'silence': self.params.get('silence', True),
            'gauss_cutoff': self.params.get('gauss_cutoff', True),
            'threshold_metric': self.params.get('threshold_metric', 'kneepoint'),
            'density_threshold': self.params.get('density_threshold', None),
            'distance_threshold': self.params.get('distance_threshold', None),
            'anormal': self.params.get('anormal', True)
        }
        
        # Create model
        model = DensityPeakCluster(**cluster_params)
        
        if self.distance_matrix is not None:
            # Use precomputed distance matrix
            labels = model.fit(self.distance_matrix, self.n_clusters)
        else:
            # Compute distance matrix from data
            from scipy.spatial.distance import pdist, squareform
            distances = pdist(X, metric='euclidean')
            distance_matrix = squareform(distances)
            labels = model.fit(distance_matrix, self.n_clusters)
        
        elapsed = time.time() - start_time
        return labels, elapsed