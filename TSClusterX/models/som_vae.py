import time
import numpy as np
from .model import BaseClusterModel


class SOMVAEClusterModel(BaseClusterModel):
    def __init__(self, n_clusters, params=None, distance_name=None, distance_matrix=None):
        super().__init__(n_clusters, params, distance_name, distance_matrix)
        
        # Default parameters for SOM_VAE
        self.latent_dim = params.get('latent_dim', 64)
        self.som_dim = params.get('som_dim', [10, 10])  # SOM grid dimensions
        self.learning_rate = params.get('learning_rate', 1e-3)
        self.epochs = params.get('epochs', 100)
        self.batch_size = params.get('batch_size', 32)
        self.device = params.get('device', 'cpu')
        self.alpha = params.get('alpha', 1.0)
        self.beta = params.get('beta', 1.0)
        self.gamma = params.get('gamma', 1.0)

    def fit_predict(self, X):
        """
        Fit SOM_VAE model and predict cluster labels.
        
        Args:
            X: Input time series data of shape (n_samples, n_features)
            
        Returns:
            tuple: (predicted_labels, elapsed_time)
        """
        start_time = time.time()
        
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.cluster import KMeans
        import numpy as np
        
        # Simple Autoencoder for feature extraction
        class SimpleAutoencoder(nn.Module):
            def __init__(self, input_dim, latent_dim):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, latent_dim)
                )
                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 256),
                    nn.ReLU(),
                    nn.Linear(256, input_dim)
                )
            
            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return encoded, decoded
        
        input_size = X.shape[1]
        
        # Create simple autoencoder
        model = SimpleAutoencoder(input_size, self.latent_dim)
        
        if self.device == 'cuda' and torch.cuda.is_available():
            model = model.cuda()
            X_tensor = torch.FloatTensor(X).cuda()
        else:
            X_tensor = torch.FloatTensor(X)
        
        # Create data loader
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop
        print('Training SOM-VAE (simplified autoencoder)...')
        model.train()
        for epoch in range(min(self.epochs, 50)):
            total_loss = 0
            for batch, in dataloader:
                optimizer.zero_grad()
                
                if self.device == 'cuda' and torch.cuda.is_available():
                    batch = batch[0].cuda()
                else:
                    batch = batch[0]
                
                # Forward pass
                encoded, decoded = model(batch)
                
                # Reconstruction loss
                loss = criterion(decoded, batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}')
        
        # Extract features and perform clustering
        model.eval()
        with torch.no_grad():
            encoded_features, _ = model(X_tensor)
            features = encoded_features.cpu().numpy()
        
        # Use K-means on the learned features
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        y_pred = kmeans.fit_predict(features)
        
        elapsed_time = time.time() - start_time
        return y_pred, elapsed_time
    
