# Models
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.parametrizations import orthogonal


class AE(nn.Module):
    def __init__(self, input_dim: int, l_space_dim: int):
        super(AE, self).__init__()
        # encoder layers
        self.enc1 = nn.Linear(input_dim, l_space_dim)
        # decoder layers
        self.dec1 = nn.Linear(l_space_dim, input_dim)

    def forward(self, x):
        # encoder-decoder
        x_enc = self.enc1(x)
        x_dec = self.dec1(x_enc)

        return x_dec, x_enc


class MixModel(nn.Module):
    def __init__(self, input_dim: int, l_space_dim: int, activation=None):
        super(MixModel, self).__init__()
        if activation is None:
            self.activation = nn.Identity()
        else:
            self.activation = activation
        # encoder layers
        self.enc1 = nn.Linear(input_dim, l_space_dim)
        # decoder layers
        self.dec1 = nn.Linear(l_space_dim, input_dim)
        # prediction
        self.fc1 = nn.Linear(l_space_dim, 1)

    def forward(self, x):
        # encoder-decoder
        x_enc = self.enc1(x)
        x_dec = self.dec1(x_enc)

        pred = self.activation(self.fc1(x_enc))

        return x_dec, x_enc, pred


class LinearModel(nn.Module):
    def __init__(self, input_dim: int, activation=None):
        super(LinearModel, self).__init__()
        if activation is None:
            self.activation = nn.Identity()
        else:
            self.activation = activation

        self.fc1 = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = self.activation(self.fc1(x))

        return x


class GaussianDropout(nn.Module):
    def __init__(self, sigma=0.1):
        super().__init__()
        self.sigma = sigma  # Standard deviation of the Gaussian noise
    
    def forward(self, x):
        if self.training:
            # Sample noise from N(1, σ²) (same shape as x)
            noise = torch.randn_like(x) * self.sigma + 1.0
            return x * noise
        else:
            return x  # Identity at test time


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, beta=1.0):
        super(VAE, self).__init__()
        
        self.beta = beta
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Linear(64, 32),
        )
        
        # Latent space parameters
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_var = nn.Linear(32, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.Linear(32, 64),
            nn.Linear(64, input_dim)
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, data):
        x = data[0]
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss(self, model_output, data, beta=1.0):
        # Reconstruction loss (MSE)
        BCE = nn.functional.mse_loss(model_output[0], data[0], reduction='sum')
        # KL divergence
        KLD = -0.5 * torch.sum(1 + model_output[2] - model_output[1].pow(2) - model_output[2].exp())

        return BCE + beta * KLD


class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)
