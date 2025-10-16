# Full Model definition with the following structure:
# - VAE on genomics and metabolomics
# - z_hat from vae is input to the time expansion plust additional features
# - transformer
# - output prediction over time
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.vae_attention.modules.transformer import TransformerEncoderLayerWithWeights
from src.vae_attention.modules.sinusoidal_position_encoder import SinusoidalPositionalEncoding
from src.vae_attention.modules.vae import VAE


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
        input_dim_genes: int,
        input_dim_metab: int,
        input_patient_features_dim: int,
        n_timepoints: int,
        model_config: dict,
        use_sampling_in_vae = None
    """
    def __init__(
        self,
        input_dim_genes: int,
        input_dim_metab: int,
        input_patient_features_dim: int,
        n_timepoints: int,
        model_config: dict,
        use_sampling_in_vae = None
    ):
        super(DeltaTimeAttentionVAE, self).__init__()

        self.input_dim_genes = input_dim_genes
        self.input_dim_metab = input_dim_metab
        self.input_patient_features_dim = input_patient_features_dim
        self.n_timepoints = n_timepoints
        self.model_config = model_config
        self.use_sampling_in_vae = use_sampling_in_vae
        
        # variables updated at each iteration of the training
        self.batch_size = 0

        # Generate causal mask
        self.causal_mask = self.generate_causal_mask(n_timepoints)

        # ------------- VAE genomics -------------
        self.vae_genes = VAE(
            input_dim=input_dim_genes,
            vae_input_to_latent_dim=model_config["vae_genomics_input_to_latent_dim"],
            vae_latent_dim=model_config["vae_genomics_latent_dim"],
            dropout=0.0
        )

        # ------------- VAE metabolomics -------------
        self.vae_metab = VAE(
            input_dim=input_dim_metab,
            vae_input_to_latent_dim=model_config["vae_metabolomics_input_to_latent_dim"],
            vae_latent_dim=model_config["vae_metabolomics_latent_dim"],
            dropout=0.0
        )

        self.film_generator_nn = nn.Sequential(
            nn.Linear(input_patient_features_dim, 2 * model_config["transformer_input_dim"]), # Outputs γ and β concatenated
            # nn.GELU()
        )

        # ---- non-linear projection of [X, y_t0] to common input dimension -----
        # Adding +1 to input_dim to account for the baseline value of y: y_t0
        tot_dim_h = model_config["vae_genomics_latent_dim"] + model_config["vae_metabolomics_latent_dim"] + 1
        self.projection_to_transformer = nn.Sequential(
            nn.Linear(tot_dim_h, model_config["transformer_input_dim"]),
            nn.GELU(),
            nn.Dropout(model_config["dropout"])
        )

        # ---------------- Time Embeddings ----------------
        self.pos_encoder = SinusoidalPositionalEncoding(
            model_config["transformer_input_dim"],
            model_config["max_len_position_enc"]
        )

        # -------------- Time-Aware transformer -------------
        self.transformer_module = TransformerEncoderLayerWithWeights(
            input_dim=model_config["transformer_input_dim"],
            nheads=model_config["n_heads"],
            dim_feedforward=model_config["transformer_dim_feedforward"],
            dropout_attention=model_config["dropout_attention"],
            dropout=model_config["dropout"]
        )

        # ------------- Final output layer -------------
        self.dropout = nn.Dropout(model_config["dropout"])
        self.fc_out = nn.Linear(model_config["transformer_input_dim"], 1)  # Predicts 1 value per timepoint
    
    def generate_causal_mask(self, T):
        return torch.triu(torch.ones(T, T) * float('-inf'), diagonal=1)

    def process_batch(self, batch):
        """
        This method has to be implemented for each specific input batch

        Args:
            batch (list): batch input(s) and target
        """
        # Input features
        x_genes = batch[0]
        x_metab = batch[1]
        patients_static_features = batch[2]
        y_baseline = batch[3]

        return x_genes, x_metab, y_baseline, patients_static_features

    def film_generator(self, x, h_in):
        film_params = self.film_generator_nn(x)
        gamma, beta = torch.chunk(film_params, 2, dim=-1) # split in half and half
        # the time dimension is always second last
        num_middle_dims = h_in.dim() - gamma.dim() # number of middle dimensions

        # Add the required number of singleton dimensions in the middle
        if num_middle_dims > 0:
            for _ in range(num_middle_dims):
                gamma = gamma.unsqueeze(-2) # Keep adding at position 1
                beta = beta.unsqueeze(-2)
        
        h_mod = gamma * h_in + beta

        return h_mod

    def expand_input_in_time(self, h):

        # Expand the static features over the time dimension
        time_dim = h.dim() - 1
        new_shape = list(h.shape)
        new_shape.insert(time_dim, self.n_timepoints)  # Insert T at the correct position
        h_exp = h.unsqueeze(time_dim).expand(new_shape)

        return h_exp
    
    def outcome_prediction(self, h):
        y_hat = self.fc_out(self.dropout(h)).squeeze(-1)  # [batch_size, ..., T]
        return y_hat

    def forward(self, batch):
        # ------------------ process input batch ------------------
        x_genes, x_metab, y_baseline, patients_static_features = self.process_batch(batch)

        # ---------------------------- VAE ----------------------------
        vae_genes_out = self.vae_genes(x_genes, use_sampling=self.use_sampling_in_vae)
        vae_metab_out = self.vae_metab(x_metab, use_sampling=self.use_sampling_in_vae)
        
        # --------------------- Concat Static fatures ----------------------
        h = torch.cat([vae_genes_out[1], vae_metab_out[1], y_baseline], dim=-1)

        # ----------- positional encoding and projection -----------
        h_exp = self.expand_input_in_time(h)

        # --------------- Projection to transformer input dimension -----------
        h_in = self.projection_to_transformer(h_exp)

        # -------- Generate FiLM parameters γ and β from static patient features --------
        h_mod = self.film_generator(patients_static_features, h_in)

        # --------------------- Time positional embedding ---------------------
        h_time = self.pos_encoder(h_mod)

        # ----------------------- Transformer ------------------------------
        h_out = self.transformer_module(h_time, attn_mask=self.causal_mask)

        # --------------------- Predict outcomes ---------------------
        y_hat = self.outcome_prediction(h_out)

        return vae_genes_out, vae_metab_out, y_hat

    def loss(self, m_out, batch):
        """
        Loss function. The structure depends on the batch data.
        To be modified according to the data used.

        VAE loss + Prediction Loss (here MSE)
        """
        x_genes, x_metab, y_baseline, patients_static_features = self.process_batch(batch)
        # vae output: x_hat, mu, logvar, z_hat
        # Reconstruction loss (MSE)
        genes_vae_loss = nn.functional.mse_loss(m_out[0][0], x_genes, reduction='sum')
        metab_vae_loss = nn.functional.mse_loss(m_out[1][0], x_metab, reduction='sum')
        # KL divergence
        genes_KLD = -0.5 * torch.sum(1 + m_out[0][2] - m_out[0][1].pow(2) - m_out[0][2].exp())
        metab_KLD = -0.5 * torch.sum(1 + m_out[1][2] - m_out[1][1].pow(2) - m_out[1][2].exp())
        # label prediction loss
        PredMSE = nn.functional.mse_loss(m_out[-1], batch[-1], reduction='sum')

        return genes_vae_loss + metab_vae_loss + genes_KLD + metab_KLD + PredMSE
    
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
            x_genes, x_metab, y_baseline, patients_static_features = self.process_batch(batch)

            # ---------------------------- VAE ----------------------------
            vae_genes_out = self.vae_genes(x_genes, use_sampling=self.use_sampling_in_vae)
            vae_metab_out = self.vae_metab(x_metab, use_sampling=self.use_sampling_in_vae)
            
            # --------------------- Concat Static fatures ----------------------
            h = torch.cat([vae_genes_out[1], vae_metab_out[1], y_baseline], dim=-1)

            # ------ positional encoding and projection ------
            h_exp = self.expand_input_in_time(h)

            # --------------- Projection to transformer input dimension -----------
            h_in = self.projection_to_transformer(h_exp)

            # -------- Generate FiLM parameters γ and β from static patient features --------
            h_mod = self.film_generator(patients_static_features, h_in)

            # --------------------- Time positional embedding ---------------------
            h_time = self.pos_encoder(h_mod)

            attn_weights = self.transformer_module.cross_attn.get_attention_weights(
                h_time,
                h_time,
                attn_mask=self.causal_mask
            )
        
        return attn_weights
