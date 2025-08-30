import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from .model import BaseClusterModel


class GNNLayer(Module):
    """Graph Neural Network Layer for SDCN"""
    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adj, active=True):
        support = torch.mm(features, self.weight)
        output = torch.mm(adj, support)  # Using dense matrix multiplication instead of sparse
        if active:
            output = F.relu(output)
        return output


class AE(nn.Module):
    """Autoencoder for SDCN"""
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3, n_input, n_z):
        super(AE, self).__init__()
        self.enc_1 = nn.Linear(n_input, n_enc_1)
        self.enc_2 = nn.Linear(n_enc_1, n_enc_2) 
        self.enc_3 = nn.Linear(n_enc_2, n_enc_3)
        self.z_layer = nn.Linear(n_enc_3, n_z)

        self.dec_1 = nn.Linear(n_z, n_dec_1)
        self.dec_2 = nn.Linear(n_dec_1, n_dec_2)
        self.dec_3 = nn.Linear(n_dec_2, n_dec_3)
        self.x_bar_layer = nn.Linear(n_dec_3, n_input)

    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        z = self.z_layer(enc_h3)

        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, enc_h1, enc_h2, enc_h3, z


class SDCN(nn.Module):
    """Structural Deep Clustering Network"""
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3, n_input, n_z, n_clusters, v=1):
        super(SDCN, self).__init__()

        # autoencoder for intra information
        self.ae = AE(n_enc_1=n_enc_1, n_enc_2=n_enc_2, n_enc_3=n_enc_3,
                     n_dec_1=n_dec_1, n_dec_2=n_dec_2, n_dec_3=n_dec_3,
                     n_input=n_input, n_z=n_z)
        
        # Try to load pretrained weights if available
        try:
            pretrain_path = './pretrain.pkl'
            if torch.cuda.is_available():
                self.ae.load_state_dict(torch.load(pretrain_path), strict=False)
            else:
                self.ae.load_state_dict(torch.load(pretrain_path, map_location='cpu'), strict=False)
        except:
            # If pretrained weights are not available, use random initialization
            pass

        # GCN for inter information
        self.gnn_1 = GNNLayer(n_input, n_enc_1)
        self.gnn_2 = GNNLayer(n_enc_1, n_enc_2)
        self.gnn_3 = GNNLayer(n_enc_2, n_enc_3)
        self.gnn_4 = GNNLayer(n_enc_3, n_z)
        self.gnn_5 = GNNLayer(n_z, n_clusters)

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        # degree
        self.v = v

    def forward(self, x, adj):
        # DNN Module
        x_bar, tra1, tra2, tra3, z = self.ae(x)

        # GCN Module
        h = self.gnn_1(x, adj)
        h = self.gnn_2(h, adj)
        h = self.gnn_3(h, adj)
        h = self.gnn_4(h, adj)
        h = self.gnn_5(h, adj, active=False)
        predict = F.softmax(h, dim=1)

        # Dual Self-supervised Module
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return x_bar, q, predict, z


class SDCNClusterModel(BaseClusterModel):
    def __init__(self, n_clusters, params=None, distance_name=None, distance_matrix=None):
        super().__init__(n_clusters, params, distance_name, distance_matrix)
        
        # Default parameters for SDCN
        self.n_enc_1 = params.get('n_enc_1', 500) if params else 500
        self.n_enc_2 = params.get('n_enc_2', 500) if params else 500
        self.n_enc_3 = params.get('n_enc_3', 2000) if params else 2000
        self.n_dec_1 = params.get('n_dec_1', 2000) if params else 2000
        self.n_dec_2 = params.get('n_dec_2', 500) if params else 500
        self.n_dec_3 = params.get('n_dec_3', 500) if params else 500
        self.n_z = params.get('n_z', 10) if params else 10
        self.lr = params.get('lr', 1e-3) if params else 1e-3
        self.epochs = params.get('epochs', 400) if params else 400
        self.device = params.get('device', 'cpu') if params else 'cpu'

    def fit_predict(self, X):
        """
        Fit SDCN model and predict cluster labels.
        
        Args:
            X: Input time series data of shape (n_samples, n_features)
            
        Returns:
            tuple: (predicted_labels, elapsed_time)
        """
        start_time = time.time()
        
        # Import sklearn here to avoid import errors if not available
        from sklearn.neighbors import kneighbors_graph
        
        # Create adjacency graph for SDCN (using k-nearest neighbors)
        k = min(10, X.shape[0] - 1)  # Use k=10 or smaller if needed
        adj = kneighbors_graph(X, n_neighbors=k, include_self=False)
        adj = adj.toarray()
        adj = adj + adj.T  # Make symmetric
        adj = np.where(adj > 0, 1, 0)  # Binarize
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        adj_tensor = torch.FloatTensor(adj)
        
        if self.device == 'cuda' and torch.cuda.is_available():
            X_tensor = X_tensor.cuda()
            adj_tensor = adj_tensor.cuda()
        
        # Adjust dimensions based on input data
        n_input = X.shape[1]
        
        # Initialize SDCN model
        model = SDCN(500, 500, 2000, 2000, 500, 500, n_input, self.n_z, 
                    self.n_clusters, v=1.0)
        
        if self.device == 'cuda' and torch.cuda.is_available():
            model = model.cuda()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        
        # Train the model
        model.train()
        print('Training SDCN...')
        for epoch in range(min(self.epochs, 100)):  # Limit epochs for efficiency
            optimizer.zero_grad()
            
            x_bar, q, pred, z = model(X_tensor, adj_tensor)
            
            # Reconstruction loss
            re_loss = torch.nn.MSELoss()(x_bar, X_tensor)
            
            # KL divergence loss (simplified)
            kl_loss = torch.nn.KLDivLoss()(q.log(), torch.softmax(pred, dim=1))
            
            loss = re_loss + 0.1 * kl_loss
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
        
        # Get predictions
        model.eval()
        with torch.no_grad():
            _, _, pred, _ = model(X_tensor, adj_tensor)
            y_pred = torch.argmax(pred, dim=1).cpu().numpy()
        
        elapsed_time = time.time() - start_time
        return y_pred, elapsed_time
        
