# Full Model definition with the following structure:
# - FFN on genomics and metabolomics
# - only attention layer
# - output prediction over time
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modules.multi_head_attention_layer import MultiHeadCrossAttentionWithWeights
from src.modules.sinusoidal_position_encoder import SinusoidalPositionalEncoding


class Model(nn.Module):
    """
    Model for longitudinal data with repeated measurements over time and 
    possibly additional dimensions (for example multiple interventions).
    This model is for static input features, i.e. covariates measured ONLY at baseline.

    The input data is expected to have the following shape:
        - Input features X: (n, ..., p)
        - Output target y: (n, ..., T)
    
    This class define a NN model with the following components:
    - FFN
    - Temporal encoding
    - Self-Attention

    Args:
        input_dim_genes: int,
        input_dim_metab: int,
        input_patient_features_dim: int,
        n_timepoints: int,
        model_config: dict
    """
    def __init__(
        self,
        input_dim_genes: int,
        input_dim_metab: int,
        input_patient_features_dim: int,
        n_timepoints: int,
        model_config: dict
    ):
        super(Model, self).__init__()

        self.input_dim_genes = input_dim_genes
        self.input_dim_metab = input_dim_metab
        self.input_patient_features_dim = input_patient_features_dim
        self.n_timepoints = n_timepoints
        self.model_config = model_config
        
        # variables updated at each iteration of the training
        self.batch_size = 0

        # Generate causal mask
        self.causal_mask = self.generate_causal_mask(n_timepoints)

        # ------------- FFN genomics -------------
        self.ffn_genes = nn.Sequential(
            nn.Linear(input_dim_genes, model_config["genomics_input_to_latent_dim"]),
            nn.GELU(),
            nn.Dropout(model_config["dropout"]),
            nn.Linear(model_config["genomics_input_to_latent_dim"], model_config["genomics_latent_dim"])
        )

        # ------------- FFN metabolomics -------------
        self.ffn_metab = nn.Sequential(
            nn.Linear(input_dim_metab, model_config["metabolomics_input_to_latent_dim"]),
            nn.GELU(),
            nn.Dropout(model_config["dropout"]),
            nn.Linear(model_config["metabolomics_input_to_latent_dim"], model_config["metabolomics_latent_dim"])
        )

        # interaction with patient features
        self.film_generator_nn = nn.Sequential(
            nn.Linear(input_patient_features_dim, 2 * model_config["attn_input_dim"]), # Outputs γ and β concatenated
            # nn.GELU()
        )

        # ---- non-linear projection of [X, y_t0] to common input dimension -----
        # Adding +1 to input_dim to account for the baseline value of y: y_t0
        tot_dim_h = model_config["genomics_latent_dim"] + model_config["metabolomics_latent_dim"] + 1
        self.projection_to_transformer = nn.Sequential(
            nn.Linear(tot_dim_h, model_config["attn_input_dim"]),
            nn.GELU(),
            nn.Dropout(model_config["dropout"])
        )

        # ---------------- Time Embeddings ----------------
        self.pos_encoder = SinusoidalPositionalEncoding(
            model_config["attn_input_dim"],
            model_config["max_len_position_enc"]
        )

        # -------------- Time-Aware transformer -------------
        self.self_attn = MultiHeadCrossAttentionWithWeights(
            input_dim=model_config["attn_input_dim"],
            nheads=model_config["n_heads"],
            dropout_attention=model_config["dropout_attention"]
        ) 

        # ------------- Final output layer -------------
        self.dropout = nn.Dropout(model_config["dropout"])
        self.fc_out = nn.Linear(model_config["attn_input_dim"], 1)  # Predicts 1 value per timepoint
    
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
        ffn_genes_out = self.ffn_genes(x_genes)
        ffn_metab_out = self.ffn_metab(x_metab)
        
        # --------------------- Concat Static fatures ----------------------
        h = torch.cat([ffn_genes_out, ffn_metab_out, y_baseline], dim=-1)

        # ----------- positional encoding and projection -----------
        h_exp = self.expand_input_in_time(h)

        # --------------- Projection to transformer input dimension -----------
        h_in = self.projection_to_transformer(h_exp)

        # -------- Generate FiLM parameters γ and β from static patient features --------
        h_mod = self.film_generator(patients_static_features, h_in)

        # --------------------- Time positional embedding ---------------------
        h_time = self.pos_encoder(h_mod)

        # ----------------------- Transformer ------------------------------
        h_out = self.self_attn(h_time, h_time, h_time, attn_mask=self.causal_mask)

        # --------------------- Predict outcomes ---------------------
        y_hat = self.outcome_prediction(h_out)

        return ffn_genes_out, ffn_metab_out, y_hat

    def loss(self, m_out, batch, reduction='sum'):
        """
        Loss function. The structure depends on the batch data.
        To be modified according to the data used.

        Only Prediction Loss (here MSE)
        """
        x_genes, x_metab, y_baseline, patients_static_features = self.process_batch(batch)

        # label prediction loss
        PredMSE = nn.functional.mse_loss(m_out[-1], batch[-1], reduction=reduction)

        loss_output = [
            PredMSE
        ]
        return loss_output
    
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
            ffn_genes_out = self.ffn_genes(x_genes)
            ffn_metab_out = self.ffn_metab(x_metab)
            
            # --------------------- Concat Static fatures ----------------------
            h = torch.cat([ffn_genes_out, ffn_metab_out, y_baseline], dim=-1)

            # ----------- positional encoding and projection -----------
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
