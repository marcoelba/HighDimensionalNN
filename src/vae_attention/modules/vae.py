# VAE module
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    """
    Transformer module that returns attention weights.
    
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
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, attn_mask=None):
        mu, logvar = self.encode(x)
        z_hat = self.reparameterize(mu, logvar)
        x_hat = self.decode(z_hat)
        
        return x_hat, mu, logvar
