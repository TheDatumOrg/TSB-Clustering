import time
import numpy as np
import warnings
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from models.model import BaseClusterModel

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. ClusterGAN will use a simplified implementation.")


class ClusterGANClusterModel(BaseClusterModel):
    """
    ClusterGAN clustering model implementation based on:
    "ClusterGAN : Latent Space Clustering in Generative Adversarial Networks"
    by Sudipto Mukherjee, Himanshu Asnani, Eugene Lin and Sreeram Kannan.
    
    This implementation follows the factory design pattern used in the TSClusterX framework.
    """
    
    def __init__(self, n_clusters, params=None, distance_name=None, distance_matrix=None):
        super().__init__(n_clusters, params, distance_name, distance_matrix)
        
        # Default parameters for ClusterGAN
        self.default_params = {
            'dim_gen': 30,  # Dimension of continuous latent code
            'beta_n': 10.0,  # Weight for generator cycle consistency loss
            'beta_c': 10.0,  # Weight for categorical cycle consistency loss
            'batch_size': 64,
            'learning_rate': 1e-4,
            'num_epochs': 100,
            'n_critics': 5,  # Number of discriminator updates per generator update
            'device': 'cpu',
            'random_state': 42,
            'verbose': True
        }
        
        # Merge with user provided parameters
        if self.params:
            self.default_params.update({k: v for k, v in self.params.items() if v is not None})
        
        self.dim_gen = self.default_params['dim_gen']
        self.beta_n = self.default_params['beta_n']
        self.beta_c = self.default_params['beta_c']
        self.batch_size = self.default_params['batch_size']
        self.learning_rate = self.default_params['learning_rate']
        self.num_epochs = self.default_params['num_epochs']
        self.n_critics = self.default_params['n_critics']
        self.device = self.default_params['device']
        self.random_state = self.default_params['random_state']
        self.verbose = self.default_params['verbose']
        
        # Set random seeds for reproducibility
        np.random.seed(self.random_state)
        if TORCH_AVAILABLE:
            torch.manual_seed(self.random_state)
    
    def fit_predict(self, X):
        """
        Fit the ClusterGAN model and predict cluster labels.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            labels: Cluster labels for each sample
            elapsed: Time taken for clustering
        """
        print(f"Using ClusterGAN with parameters: {self.default_params}")
        start_time = time.time()

        # Normalize input data
        X_normalized = self._normalize_data(X)
        
        if self.verbose:
            print(f"Input data shape: {X.shape}")
            print(f"Number of clusters: {self.n_clusters}")
            print(f"Generator latent dimension: {self.dim_gen}")
        
        # Initialize networks
        input_dim = X.shape[1]
        z_dim = self.dim_gen + self.n_clusters  # Continuous + categorical dimensions
        
        self.generator = self._create_generator(z_dim, input_dim)
        self.discriminator = self._create_discriminator(input_dim)
        self.encoder = self._create_encoder(input_dim, self.dim_gen, self.n_clusters)
        
        # Train the ClusterGAN model
        self._train_clustergan(X_normalized)
        
        # Extract cluster assignments from trained model
        labels = self._extract_clusters(X_normalized)
        
        elapsed = time.time() - start_time
        
        if self.verbose:
            print(f"ClusterGAN training completed in {elapsed:.2f} seconds")
        
        return labels, elapsed
    
    def _normalize_data(self, X):
        """Normalize input data to [0, 1] range."""
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        X_range = X_max - X_min
        X_range[X_range == 0] = 1  # Avoid division by zero
        return (X - X_min) / X_range
    
    def _create_generator(self, z_dim, output_dim):
        """Create generator network."""
        class Generator(nn.Module):
            def __init__(self, z_dim, output_dim):
                super(Generator, self).__init__()
                self.network = nn.Sequential(
                    nn.Linear(z_dim, 128),
                    nn.ReLU(inplace=True),
                    nn.Linear(128, 256),
                    nn.ReLU(inplace=True),
                    nn.Linear(256, 512),
                    nn.ReLU(inplace=True),
                    nn.Linear(512, output_dim),
                    nn.Sigmoid()
                )
            
            def forward(self, z):
                return self.network(z)
        
        return Generator(z_dim, output_dim).to(self.device)
    
    def _create_discriminator(self, input_dim):
        """Create discriminator network."""
        class Discriminator(nn.Module):
            def __init__(self, input_dim):
                super(Discriminator, self).__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_dim, 512),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Linear(512, 256),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Linear(256, 128),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Linear(128, 1)
                )
            
            def forward(self, x):
                return self.network(x)
        
        return Discriminator(input_dim).to(self.device)
    
    def _create_encoder(self, input_dim, dim_gen, n_clusters):
        """Create encoder network."""
        class Encoder(nn.Module):
            def __init__(self, input_dim, dim_gen, n_clusters):
                super(Encoder, self).__init__()
                self.shared = nn.Sequential(
                    nn.Linear(input_dim, 512),
                    nn.ReLU(inplace=True),
                    nn.Linear(512, 256),
                    nn.ReLU(inplace=True),
                    nn.Linear(256, 128),
                    nn.ReLU(inplace=True)
                )
                
                # Continuous latent code branch
                self.gen_branch = nn.Linear(128, dim_gen)
                
                # Categorical latent code branch
                self.cat_branch = nn.Linear(128, n_clusters)
            
            def forward(self, x):
                shared_out = self.shared(x)
                gen_out = self.gen_branch(shared_out)
                cat_logits = self.cat_branch(shared_out)
                cat_out = torch.softmax(cat_logits, dim=1)
                return gen_out, cat_out, cat_logits
        
        return Encoder(input_dim, dim_gen, n_clusters).to(self.device)
    
    def _sample_z(self, batch_size):
        """Sample latent codes (continuous + categorical)."""
        # Sample continuous part
        z_gen = torch.randn(batch_size, self.dim_gen, device=self.device) * 0.1
        
        # Sample categorical part (one-hot)
        z_cat_idx = torch.randint(0, self.n_clusters, (batch_size,), device=self.device)
        z_cat = torch.zeros(batch_size, self.n_clusters, device=self.device)
        z_cat.scatter_(1, z_cat_idx.unsqueeze(1), 1)
        
        # Concatenate continuous and categorical parts
        z = torch.cat([z_gen, z_cat], dim=1)
        return z, z_gen, z_cat
    
    def _train_clustergan(self, X):
        """Train the ClusterGAN model."""
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Initialize optimizers
        g_optimizer = optim.Adam(
            list(self.generator.parameters()) + list(self.encoder.parameters()),
            lr=self.learning_rate, betas=(0.5, 0.9)
        )
        d_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=self.learning_rate, betas=(0.5, 0.9)
        )
        
        # Training loop
        for epoch in range(self.num_epochs):
            for batch_idx, (real_data,) in enumerate(dataloader):
                batch_size = real_data.size(0)
                
                # Train discriminator
                for _ in range(self.n_critics):
                    d_optimizer.zero_grad()
                    
                    # Real data
                    d_real = self.discriminator(real_data)
                    
                    # Fake data
                    z, _, _ = self._sample_z(batch_size)
                    fake_data = self.generator(z)
                    d_fake = self.discriminator(fake_data.detach())
                    
                    # Wasserstein loss with gradient penalty
                    d_loss = -torch.mean(d_real) + torch.mean(d_fake)
                    d_loss += self._gradient_penalty(real_data, fake_data)
                    
                    d_loss.backward()
                    d_optimizer.step()
                
                # Train generator and encoder
                g_optimizer.zero_grad()
                
                # Generate fake data
                z, z_gen, z_cat = self._sample_z(batch_size)
                fake_data = self.generator(z)
                
                # Generator loss (fool discriminator)
                d_fake = self.discriminator(fake_data)
                g_loss = -torch.mean(d_fake)
                
                # Cycle consistency losses
                z_enc_gen, z_enc_cat, z_enc_logits = self.encoder(fake_data)
                
                # Continuous cycle consistency
                cycle_gen_loss = torch.mean(torch.square(z_gen - z_enc_gen))
                
                # Categorical cycle consistency
                cycle_cat_loss = torch.mean(
                    torch.nn.functional.cross_entropy(z_enc_logits, torch.argmax(z_cat, dim=1), reduction='none')
                )
                
                # Total generator loss
                total_g_loss = g_loss + self.beta_n * cycle_gen_loss + self.beta_c * cycle_cat_loss
                
                total_g_loss.backward()
                g_optimizer.step()
            
            if self.verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{self.num_epochs}], "
                      f"D_loss: {d_loss.item():.4f}, "
                      f"G_loss: {total_g_loss.item():.4f}")
    
    def _gradient_penalty(self, real_data, fake_data, lambda_gp=10):
        """Compute gradient penalty for WGAN-GP."""
        batch_size = real_data.size(0)
        epsilon = torch.rand(batch_size, 1, device=self.device)
        epsilon = epsilon.expand_as(real_data)
        
        interpolated = epsilon * real_data + (1 - epsilon) * fake_data
        interpolated.requires_grad_(True)
        
        d_interpolated = self.discriminator(interpolated)
        
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = lambda_gp * torch.mean((gradients.norm(2, dim=1) - 1) ** 2)
        
        return gradient_penalty
    
    def _extract_clusters(self, X):
        """Extract cluster assignments from trained encoder."""
        self.encoder.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            _, z_cat, _ = self.encoder(X_tensor)
            cluster_probs = z_cat.cpu().numpy()
        
        # Use categorical probabilities for clustering
        if self.beta_n == 0:
            # Use only categorical part
            labels = np.argmax(cluster_probs, axis=1)
        else:
            # Use K-means on the categorical probabilities
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
            labels = kmeans.fit_predict(cluster_probs)
        
        return labels
    