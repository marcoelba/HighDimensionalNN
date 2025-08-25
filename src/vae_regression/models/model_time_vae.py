import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


# VAE + regression
class TimeAwareRegVAE(nn.Module):
    def __init__(
        self,
        input_dim,          # Dimension of fixed input X (e.g., number of genes)
        latent_dim,         # Dimension of latent space Z
        n_timepoints,     # Number of timepoints (T)
        n_measurements,
        input_to_latent_dim=32,
        transformer_dim_feedforward=32,
        nhead=4,
        time_emb_dim=8,     # Dimension of time embeddings
        dropout_sigma=0.1,
        beta_vae=1.0,
        prediction_weight=1.0,
        reconstruction_weight=1.0
    ):
        super(TimeAwareRegVAE, self).__init__()
        
        self.beta = beta_vae
        self.reconstruction_weight = reconstruction_weight
        self.prediction_weight = prediction_weight
        self.n_timepoints = n_timepoints
        self.n_measurements = n_measurements
        self.nhead = nhead
        self.transformer_input_dim = input_dim + time_emb_dim

        # --- VAE (unchanged) ---
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_to_latent_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(input_to_latent_dim, latent_dim)
        self.fc_var = nn.Linear(input_to_latent_dim, latent_dim)
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_to_latent_dim),
            nn.ReLU(),
            nn.Linear(input_to_latent_dim, input_dim)
        )

        # --- Time Embeddings ---
        self.time_embedding = nn.Embedding(n_timepoints, time_emb_dim)

        # --- Latent-to-Outcome Mapping ---
        # Project latent features to a space compatible with time embeddings
        # self.latent_proj = nn.Linear(input_dim, input_to_latent_dim)  # Projects x_hat

        # --- Time-Aware Prediction Head ---
        # Lightweight Transformer
        encoder_layer = TransformerEncoderLayer(
            d_model=self.transformer_input_dim,  # Input dim
            nhead=self.nhead,                    # Number of attention heads
            dim_feedforward=transformer_dim_feedforward,
            dropout=dropout_sigma,
            activation="gelu"
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=1)
        self.fc_out = nn.Linear(self.transformer_input_dim, 1)  # Predicts 1 value per timepoint

        # Dropout
        self.dropout = nn.Dropout(dropout_sigma)

    def encode(self, x):
        x1 = self.encoder(x)
        mu = self.fc_mu(x1)
        lvar = self.fc_var(x1)
        return mu, lvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def generate_causal_mask(self, T):
        return torch.triu(torch.ones(T, T) * float('-inf'), diagonal=1)

    def generate_measurement_mask(self, batch_size, M):
        return torch.ones([batch_size, M])

    def forward(self, x):
        
        batch_size, max_meas, input_dim = x.shape

        # --- VAE Forward Pass ---
        # Generate causal mask
        causal_mask = self.generate_causal_mask(self.n_timepoints).to(x.device)

        x_flat = x.view(-1, input_dim)  # (batch_size * max_measurements, input_dim)
        mu, logvar = self.encode(x_flat)
        z_hat = self.reparameterize(mu, logvar)
        x_hat_flat = self.decode(z_hat)  # Shape: [batch_size, input_dim]
        x_hat = x_hat_flat.view(batch_size, max_meas, input_dim)  # Reshape back

        # --- Time-Aware Prediction ---
        # Project latent features
        # h = self.latent_proj(x_hat_flat)  # Shape: [batch_size*M, 32]
        # h = h.unsqueeze(1).repeat(1, self.n_timepoints, 1)  # [batch_size*M, T, 32]
        h = x_hat_flat.unsqueeze(1).repeat(1, self.n_timepoints, 1)  # [batch_size*M, T, input_dim]

        # Get time embeddings (for all timepoints)
        time_ids = torch.arange(self.n_timepoints, device=x.device)  # [0, 1, ..., T-1]
        time_embs = self.time_embedding(time_ids)  # [T, time_emb_dim]
        time_embs = time_embs.unsqueeze(0).repeat(h.shape[0], 1, 1)  # [bs*M, T, time_emb_dim]

        # Combine latent features and time embeddings
        h_time = torch.cat([h, time_embs], dim=-1)  # [batch_size*M, T, input_dim + time_emb_dim]

        # Process temporally
        # Transformer expects [T, batch_size, features]
        h_time = h_time.transpose(0, 1)  # [T, batch_size*M, ...]
        h_out = self.transformer(h_time, mask=causal_mask)  # [T, batch_size, ...]
        h_out = h_out.transpose(0, 1)    # [batch_size*M, T, ...]

        # Predict outcomes
        y_hat_flat = self.fc_out(self.dropout(h_out)).squeeze(-1)  # [batch_size*M, T]
        y_hat = y_hat_flat.view(batch_size, max_meas, self.n_timepoints)
        
        return x_hat, y_hat, mu, logvar

    def predict(self, x):

        # --- VAE Forward Pass ---
        # Generate causal mask
        causal_mask = self.generate_causal_mask(self.n_timepoints).to(x.device)

        mu, logvar = self.encode(x)
        z_hat = self.reparameterize(mu, logvar)
        x_hat_flat = self.decode(z_hat)  # Shape: [batch_size, input_dim]

        # --- Time-Aware Prediction ---
        # Project latent features
        # h = self.latent_proj(x_hat_flat)  # Shape: [batch_size*M, 32]
        # h = h.unsqueeze(1).repeat(1, self.n_timepoints, 1)  # [batch_size*M, T, 32]

        h = x_hat_flat.unsqueeze(1).repeat(1, self.n_timepoints, 1)  # [batch_size*M, T, 32]

        # Get time embeddings (for all timepoints)
        time_ids = torch.arange(self.n_timepoints, device=x.device)  # [0, 1, ..., T-1]
        time_embs = self.time_embedding(time_ids)  # [T, time_emb_dim]
        time_embs = time_embs.unsqueeze(0).repeat(h.shape[0], 1, 1)  # [bs*M, T, time_emb_dim]

        # Combine latent features and time embeddings
        h_time = torch.cat([h, time_embs], dim=-1)  # [batch_size*M, T, 32 + time_emb_dim]

        # Process temporally
        # Transformer expects [T, batch_size, features]
        h_time = h_time.transpose(0, 1)  # [T, batch_size*M, ...]
        h_out = self.transformer(h_time, mask=causal_mask)  # [T, batch_size, ...]
        h_out = h_out.transpose(0, 1)    # [batch_size*M, T, ...]

        # Predict outcomes
        y_hat_flat = self.fc_out(self.dropout(h_out)).squeeze(-1)  # [batch_size*M, T]

        return y_hat_flat

    def loss(self, m_out, x, y):
        # Reconstruction loss (MSE)
        BCE = nn.functional.mse_loss(m_out[0], x, reduction='sum')
        # KL divergence
        KLD = -0.5 * torch.sum(1 + m_out[3] - m_out[2].pow(2) - m_out[3].exp())
        # label prediction loss
        PredMSE = nn.functional.mse_loss(m_out[1], y, reduction='sum')

        return self.reconstruction_weight * BCE + self.beta * KLD + self.prediction_weight * PredMSE


def loss_components(x, y, x_hat, y_hat, mu, logvar):
    
    # Reconstruction loss (MSE)
    reconstruction_loss = nn.functional.mse_loss(x_hat, x, reduction='none')
    # KL divergence
    KLD = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    # label prediction loss
    prediction_loss = nn.functional.mse_loss(y_hat, y, reduction='none')

    return reconstruction_loss, KLD, prediction_loss
