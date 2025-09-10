# Full Model definition
import torch
import torch.nn as nn
import torch.nn.functional as F

# import os
# os.chdir("./src")

from utils.decorator import compose_docstring

from vae_attention.modules.transformer import TransformerEncoderLayerWithWeights
from vae_attention.modules.sinusoidal_position_encoder import SinusoidalPositionalEncoding
from vae_attention.modules.vae import VAE


@compose_docstring(VAE, SinusoidalPositionalEncoding, TransformerEncoderLayerWithWeights)
class DeltaTimeAttentionVAE(nn.Module):
    """
    Model for longitudinal data with repeated measurements over time and 
    possibly additional dimensions (for example multiple interventions).
    This model is for static input features, i.e. covariates measured ONLY at baseline.

    The input data is expected to have the following shape:
        - Input features X: (n, ..., p)
        - Output target y: (n, ..., T)
    
    This class define a NN model with the following components:
    - VAE
    - Temporal encoding
    - Transformer

    Args:
        beta_vae=1.0,
        prediction_weight=1.0,
        reconstruction_weight=1.0
    """
    def __init__(
        self,
        input_dim,
        n_timepoints,
        vae_latent_dim,
        vae_input_to_latent_dim,
        max_len_position_enc,
        transformer_input_dim,
        transformer_dim_feedforward,
        nheads=4,
        dropout=0.0,
        dropout_attention=0.0,
        beta_vae=1.0,
        prediction_weight=1.0,
        reconstruction_weight=1.0
    ):
        super(DeltaTimeAttentionVAE, self).__init__()
        
        # loss weights
        self.beta = beta_vae
        self.reconstruction_weight = reconstruction_weight
        self.prediction_weight = prediction_weight
        # others
        self.n_timepoints = n_timepoints
        self.nheads = nheads
        self.transformer_input_dim = transformer_input_dim
        self.input_dim = input_dim
        # variables updated at each iteration of the training
        self.batch_size = 0
        self.max_meas = 0

        # Generate causal mask
        self.causal_mask = self.generate_causal_mask(n_timepoints)

        # ------------- VAE -------------
        self.vae = VAE(
            input_dim=input_dim,
            vae_input_to_latent_dim=vae_input_to_latent_dim,
            vae_latent_dim=vae_latent_dim,
            dropout=0.0
        )

        # ---- non-linear projection of [X, y_t0] to common input dimension -----
        # Adding +1 to input_dim to account for the baseline value of y: y_t0
        self.projection_to_transformer = nn.Sequential(
            nn.Linear(input_dim + 1, transformer_input_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # ---------------- Time Embeddings ----------------
        self.pos_encoder = SinusoidalPositionalEncoding(
            transformer_input_dim,
            max_len_position_enc
        )

        # -------------- Time-Aware transformer -------------
        self.transformer_module = TransformerEncoderLayerWithWeights(
            input_dim=self.transformer_input_dim,
            nheads=self.nheads,
            dim_feedforward=transformer_dim_feedforward,
            dropout_attention=dropout_attention,
            dropout=dropout
        )

        # ------------- Final output layer -------------
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(self.transformer_input_dim, 1)  # Predicts 1 value per timepoint
    
    def generate_causal_mask(self, T):
        return torch.triu(torch.ones(T, T) * float('-inf'), diagonal=1)

    def generate_measurement_mask(self, batch_size, M):
        return torch.ones([batch_size, M])

    def preprocess_input(self, batch):
        """
        This method has to be implemented for each specific input batch

        Args:
            batch (list): batch input(s) and target 
        """
        # Input features
        x = batch[0]
        y_baseline = batch[1]
        
        return x, y_baseline

    def make_transformer_input(self, x_hat, y0):

        # --------------------- Add the lagged outcome y ----------------------
        h = torch.cat([x_hat, y0], dim=-1)

        # --------------------- Expand the static features over the time dimension ---------------------
        time_dim = h.dim() - 1
        new_shape = list(h.shape)
        new_shape.insert(time_dim, self.n_timepoints)  # Insert T at the correct position
        h_exp = h.unsqueeze(time_dim).expand(new_shape)

        # --------------- Projection to transformer input dimension -----------
        h_in = self.projection_to_transformer(h_exp)

        # --------------------- Time positional embedding ---------------------
        h_time = self.pos_encoder(h_in)

        return h_time
    
    def outcome_prediction(self, h):
        y_hat = self.fc_out(self.dropout(h)).squeeze(-1)  # [batch_size, ..., T]
        return y_hat

    def forward(self, batch):
        # ------------------ process input batch ------------------
        x, y_baseline = self.preprocess_input(batch)

        # ---------------------------- VAE ----------------------------
        x_hat, mu, logvar = self.vae(x)
        
        # ------ concatenate with y0, positional encoding and projection ------
        h_time = self.make_transformer_input(x_hat, y_baseline)

        # ----------------------- Transformer ------------------------------
        # This custom Transformer module expects input with shape: [batch_size, seq_len, input_dim]
        h_out = self.transformer_module(h_time, attn_mask=self.causal_mask)

        # --------------------- Predict outcomes ---------------------
        y_hat = self.outcome_prediction(h_out)

        return x_hat, y_hat, mu, logvar

    def loss(self, m_out, batch):
        """
        Loss function. The structure depends on the batch data.
        To be modified according to the data used.

        VAE loss + Prediction Loss (here MSE)
        """
        # Reconstruction loss (MSE)
        BCE = nn.functional.mse_loss(m_out[0], batch[0], reduction='sum')
        # KL divergence
        KLD = -0.5 * torch.sum(1 + m_out[3] - m_out[2].pow(2) - m_out[3].exp())
        # label prediction loss
        PredMSE = nn.functional.mse_loss(m_out[1], batch[2], reduction='sum')

        return self.reconstruction_weight * BCE + self.beta * KLD + self.prediction_weight * PredMSE
    
    def get_attention_weights(self, batch):
        """
        This method allows to extract the attention weights from the layer.
        attn_weights shape: [batch_size, nhead, seq_len, seq_len]
        attn_weights[0, 0] would be a [seq_len, seq_len] matrix,
        showing the attention pattern for the FIRST head for the FIRST input X.
        The element at position [i, j] answers: "For the token at position i (Query), 
        how much did it pay attention to the token at position j (Key)?

        Args:
            batch: Input batch tensors, same shape as used in training
        
        Returns:
            attn_weights: Attention weights tensor [batch_size, nhead, seq_len, seq_len]
        """
        with torch.no_grad():
            # ------------------ process input batch ------------------
            x, y_baseline = self.preprocess_input(batch)

            # ---------------------------- VAE ----------------------------
            x_hat, mu, logvar = self.vae(x)
            
            # ------ concatenate with y0, positional encoding and projection ------
            h_time = self.make_transformer_input(x_hat, y_baseline)

            attn_weights = self.transformer_module.cross_attn.get_attention_weights(
                h_time,
                h_time,
                attn_mask=self.causal_mask
            )
        
        return attn_weights
