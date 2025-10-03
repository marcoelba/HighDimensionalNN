# VAE module
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    """
    Variational Autoencoder module
    
    Args:
        input_dim (int): Input dimension for VAE.
        vae_input_to_latent_dim (int): Dimension of middle hidden layer.
        vae_latent_dim (int): Dimension of latent space.
        dropout (float, optional): Dropout probability. Default: 0.0
    """

    def __init__(self, input_dim, vae_input_to_latent_dim, vae_latent_dim, dropout=0.0):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, vae_input_to_latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.fc_mu = nn.Linear(vae_input_to_latent_dim, vae_latent_dim)
        self.fc_var = nn.Linear(vae_input_to_latent_dim, vae_latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(vae_latent_dim, vae_input_to_latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(vae_input_to_latent_dim, input_dim)
        )
    
    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        lvar = self.fc_var(x)

        return mu, lvar
    
    def reparameterize(self, mu, logvar, use_sampling=None):
        
        if use_sampling is None:
            use_sampling = self.training  # Sample during training, use mean during eval
        
        if use_sampling:
            # Sample during training for regularization
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            # Use mean during inference for stability
            z = mu

        return z
    
    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, use_sampling=None):
        mu, logvar = self.encode(x)
        z_hat = self.reparameterize(mu, logvar, use_sampling=use_sampling)
        x_hat = self.decode(z_hat)
        
        return x_hat, mu, logvar, z_hat
