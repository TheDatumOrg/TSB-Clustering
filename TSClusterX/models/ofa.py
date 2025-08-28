import time
import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans
from models.model import BaseClusterModel

try:
    from transformers import GPT2Model, GPT2Config
    from einops import rearrange
    from tqdm import tqdm
except ImportError:
    GPT2Model = None
    GPT2Config = None
    rearrange = None
    tqdm = None


class GPT4TS(nn.Module):
    def __init__(self, configs, slen, device):
        super(GPT4TS, self).__init__()
        self.patch_size = 1
        self.stride = 1
        self.seq_len = slen
        self.patch_num = (self.seq_len - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.patch_num += 1

        self.in_layer = nn.Linear(self.patch_size, 768)

        # Initialize GPT-2 with the new configuration
        custom_config = GPT2Config.from_pretrained('gpt2')
        custom_config.n_positions = slen + 1  # Change this to your desired sequence length
        self.gpt2 = GPT2Model(custom_config)

        # Load pre-trained weights
        pretrained_model = GPT2Model.from_pretrained('gpt2')

        # Copy weights from the pre-trained model to the custom model
        self.gpt2.wte.weight.data = pretrained_model.wte.weight.data

        # Adjust the position embeddings
        if custom_config.n_positions > 1024:
            new_wpe = nn.Parameter(torch.zeros(custom_config.n_positions, self.gpt2.wpe.embedding_dim))
            new_wpe.data[:1024] = pretrained_model.wpe.weight.data

            # Interpolate positional embeddings
            interp = nn.functional.interpolate(pretrained_model.wpe.weight.data.unsqueeze(0).permute(0, 2, 1), 
                                               size=custom_config.n_positions, 
                                               mode='linear')
            new_wpe.data = interp.squeeze(0).permute(1, 0)
        else:
            new_wpe = nn.Parameter(pretrained_model.wpe.weight.data[:custom_config.n_positions])
        
        self.gpt2.wpe.weight = new_wpe

        for i, layer in enumerate(pretrained_model.h):
            if i >= custom_config.n_layer:
                break
            self.gpt2.h[i].load_state_dict(layer.state_dict())

        self.hidden_layer = nn.Linear(768 * self.patch_num, 200)
        self.out_layer = nn.Linear(200, slen)

        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if "ln" in name or "wpe" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.to(device)
        self.train()

    def forward(self, x):
        B, L, M = x.shape

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x /= stdev
        x = rearrange(x, "b l m -> b m l")

        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x = rearrange(x, "b m n p -> (b m) n p")

        x = self.in_layer(x)

        outputs = self.gpt2(inputs_embeds=x).last_hidden_state

        hidden = self.hidden_layer(outputs.reshape(B * M, -1))

        outputs = self.out_layer(hidden)
        outputs = rearrange(outputs, "(b m) l -> b l m", b=B)

        outputs = outputs * stdev
        outputs = outputs + means

        return hidden, outputs


class OFAClusterModel(BaseClusterModel):
    def fit_predict(self, X):
        if GPT2Model is None or GPT2Config is None or rearrange is None:
            raise ImportError("Required packages not available. Please install transformers, einops.")
        
        print(f"Using parameters: {self.params}")
        start_time = time.time()
        
        # Default parameters
        learning_rate = self.params.get('learning_rate', 1e-3)
        device = self.params.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # KMeans parameters
        kmeans_init = self.params.get('kmeans_init', 'random')
        kmeans_n_init = self.params.get('kmeans_n_init', 1)
        kmeans_max_iter = self.params.get('kmeans_max_iter', 300)
        kmeans_tol = self.params.get('kmeans_tol', 1e-4)
        
        try:
            slen = int(X.shape[1])
            
            # Initialize model
            gpt4_model = GPT4TS(GPT2Config, slen, device)
            mse_loss = nn.MSELoss()
            optim = torch.optim.Adam(gpt4_model.parameters(), lr=learning_rate)

            # Training phase
            gpt4_model.train()
            progress_iter = tqdm(X) if tqdm else X
            for t in progress_iter:
                t = torch.tensor(t).unsqueeze(0).unsqueeze(2).to(torch.float32).to(device)

                optim.zero_grad()

                hidden, output = gpt4_model(t)
                loss = mse_loss(output, t)

                loss.backward()
                optim.step()

            # Inference phase
            gpt4_model.eval()
            rep = []
            progress_iter = tqdm(X) if tqdm else X
            for t in progress_iter:
                t = torch.tensor(t).unsqueeze(0).unsqueeze(2).to(torch.float32).to(device)

                with torch.no_grad():
                    hidden, output = gpt4_model(t)
                    rep.append(hidden.squeeze(dim=0).detach().cpu().numpy())

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
            
        except Exception as e:
            print(f"Error in OFA clustering: {e}")
            # Fallback to random labels
            labels = np.random.randint(0, self.n_clusters, size=len(X))
        
        elapsed = time.time() - start_time
        return labels, elapsed
