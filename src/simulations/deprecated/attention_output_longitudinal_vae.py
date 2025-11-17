# Multi measurements longitudinal vae-mlp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

import numpy as np
from scipy.stats import pearsonr

import matplotlib.pyplot as plt
import shap

import os
os.chdir("./src")

from vae_regression import data_generation
from vae_regression import training
from vae_regression.models.multi_head_attention_layer import MultiHeadSelfAttentionWithWeights, TransformerEncoderLayerWithWeights
from vae_regression.models.sinusoidal_position_encoder import SinusoidalPositionalEncoding

from model_utils import utils


torch.get_num_threads()
torch.set_num_threads(6)

# generate data assuming an underlying latent space of dimension k
k = 5
n_train = 100
n_test = 200
n_val = 200
n = n_train + n_val + n_test
p = 600
p1 = 30
p0 = p - p1
n_timepoints = 5
n_measurements = 4
batch_size = 50

# custom W
np.random.seed(323425)
W = np.random.choice(
    [-1.5, -1, -0.8, -0.5, 1.5, 1, 0.8, 0.5],
    size=(k, p)
)
first_half = range(0, int(p / 2))
second_half = range(int(p / 2), p)
# # block structure
# W[0, first_half] = 0.0
# W[1, first_half] = 0.0
# W[3, second_half] = 0.0
# W[4, second_half] = 0.0
# # first p0 features do NOT have any effect
W[:, 0:p0] = 0

beta = np.array([-1, 1, -1, 1, -1])
beta = beta[..., None] * np.ones([k, n_timepoints])
beta[0, 1:] = [-2., -2., -1., -1]
beta[1, 1:] = [2., 3., 1., 1]
beta[2, 1:] = [0, 0, 0, 0]

beta_time = np.array([0, 1, 2, 1, 0, -1])

y, X, Z, beta = data_generation.multi_longitudinal_data_generation(
    n, k, p, n_timepoints, n_measurements,
    noise_scale = 0.5,
    W=W,
    beta=beta,
    beta_time=beta_time
)

# one measurement
plt.plot(y[0:5, 0, :].transpose())
plt.show()

# one patient
plt.plot(y[0, :, :].transpose())
plt.show()

# get tensors
X_tensor = torch.FloatTensor(X).to(torch.device("cpu"))
Z_tensor = torch.FloatTensor(Z).to(torch.device("cpu"))
y_tensor = torch.FloatTensor(y).to(torch.device("cpu"))

# split data
data_split = utils.DataSplit(X.shape[0], test_size=n_test, val_size=n_val)
print("train: ", len(data_split.train_index), 
    "val: ", len(data_split.val_index),
    "test: ", len(data_split.test_index)
)


data_train = data_split.get_train(X, Z, y)
data_test = data_split.get_test(X, Z, y)
data_val = data_split.get_val(X, Z, y)

tensor_data_train = data_split.get_train(X_tensor, y_tensor)
tensor_data_test = data_split.get_test(X_tensor, y_tensor)
tensor_data_val = data_split.get_val(X_tensor, y_tensor)

# make tensor data loaders
train_dataloader = utils.make_data_loader(*tensor_data_train, batch_size=batch_size)
test_dataloader = utils.make_data_loader(*tensor_data_test, batch_size=batch_size)
val_dataloader = utils.make_data_loader(*tensor_data_val, batch_size=batch_size)

next(iter(train_dataloader))[0].shape  # X
next(iter(train_dataloader))[1].shape  # y


def plot_attention_weights(attn_weights, observation, layer_name="Attention"):
    """
    Plot attention weights for all heads in a grid.
    
    Args:
        attn_weights: Tensor of shape [batch_size, nhead, seq_len, seq_len]
        observation: Observation to plot
        layer_name: Name for the plot title
    """
    batch_size, nhead, seq_len, _ = attn_weights.shape
    
    # Use weights from first batch element
    weights = attn_weights[observation].detach().cpu().numpy()
    
    # Create subplot grid
    fig, axes = plt.subplots(1, nhead, figsize=(4 * nhead, 4))
    if nhead == 1:
        axes = [axes]  # Make it iterable
    
    for i, ax in enumerate(axes):
        im = ax.imshow(weights[i], cmap='viridis', aspect='auto', vmin=0, vmax=1)
        ax.set_title(f'Head {i+1}')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        plt.colorbar(im, ax=ax)
    
    plt.suptitle(f'{layer_name} Weights (Batch 0)')
    plt.tight_layout()
    plt.show()


# VAE + Explicit multi-head Attention layer
class TimeAttentionVAE(nn.Module):
    """
        Define a NN model for longitudinal data and repeated measurements.
        Uses VAE and Attention layers.
        Takes in input a set of features measuted at baseline and the known outcome also measured at baseline.
        Predicts T time points in the future, i.e. the trajectory of the outcome of interest.
    """
    def __init__(
        self,
        input_dim,          # Dimension of fixed input X (e.g., number of genes)
        n_timepoints,     # Number of timepoints (T)
        n_measurements,
        vae_latent_dim,         # Dimension of latent space Z in VAE
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
        super(TimeAttentionVAE, self).__init__()
        
        self.beta = beta_vae
        self.reconstruction_weight = reconstruction_weight
        self.prediction_weight = prediction_weight
        self.n_timepoints = n_timepoints
        self.n_measurements = n_measurements
        self.nheads = nheads
        self.transformer_input_dim = transformer_input_dim

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # ------------- VAE -------------
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, vae_input_to_latent_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(vae_input_to_latent_dim, vae_latent_dim)
        self.fc_var = nn.Linear(vae_input_to_latent_dim, vae_latent_dim)
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(vae_latent_dim, vae_input_to_latent_dim),
            nn.ReLU(),
            nn.Linear(vae_input_to_latent_dim, input_dim)
        )

        # ---- non-linear projection of [X,lag_y] to common input dimension -----
        # X already extended to the time dimension
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
        self.fc_out = nn.Linear(self.transformer_input_dim, 1)  # Predicts 1 value per timepoint

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

    def forward(self, batch):
        x = batch[0]
        # make lagged y
        past_y = torch.zeros(batch[1].shape)
        past_y = past_y[..., None]
        past_y[:, :, 1:, 0] = batch[1][:, :, :-1]  # Fill positions t>=2 with y values from t-1
        past_y_flat = past_y.view(-1, self.n_timepoints, 1)

        batch_size, max_meas, input_dim = x.shape
        # Generate causal mask
        causal_mask = self.generate_causal_mask(self.n_timepoints).to(x.device)

        # ---------------------------- VAE ----------------------------
        x_flat = x.view(-1, input_dim)  # (batch_size * max_measurements, input_dim)
        mu, logvar = self.encode(x_flat)
        z_hat = self.reparameterize(mu, logvar)
        x_hat_flat = self.decode(z_hat)  # Shape: [batch_size, input_dim]
        x_hat = x_hat_flat.view(batch_size, max_meas, input_dim)  # Reshape back

        # --------------------- Expand over time dimension ---------------------
        h = x_hat_flat.unsqueeze(1).repeat(1, self.n_timepoints, 1)  # [batch_size*M, T, input_dim]

        # --------------------- Add the lagged outcome y ----------------------
        h_with_lag_y = torch.cat([h, past_y_flat], dim=-1)

        # --------------- Projection to transformer input dimension -----------
        h_in = self.projection_to_transformer(h_with_lag_y)
        
        # --------------------- Time positional embedding ---------------------
        h_time = self.pos_encoder(h_in)

        # ----------------------- Transformer ------------------------------
        # This custom Transformer module expects input with shape: [batch_size, seq_len, input_dim]
        h_out = self.transformer_module(h_time, attn_mask=causal_mask)

        # Predict outcomes
        y_hat_flat = self.fc_out(self.dropout(h_out)).squeeze(-1)  # [batch_size*M, T]
        y_hat = y_hat_flat.view(batch_size, max_meas, self.n_timepoints)
        
        return x_hat, y_hat, mu, logvar

    def loss(self, m_out, x, y):
        # Reconstruction loss (MSE)
        BCE = nn.functional.mse_loss(m_out[0], x, reduction='sum')
        # KL divergence
        KLD = -0.5 * torch.sum(1 + m_out[3] - m_out[2].pow(2) - m_out[3].exp())
        # label prediction loss
        PredMSE = nn.functional.mse_loss(m_out[1], y, reduction='sum')

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
            x: Input tensors, same shape as used in training
        
        Returns:
            attn_weights: Attention weights tensor [batch_size, nhead, seq_len, seq_len]
        """
        with torch.no_grad():
            x = batch[0]
            # make lagged y
            past_y = torch.zeros(batch[1].shape)
            past_y = past_y[..., None]
            past_y[:, :, 1:, 0] = batch[1][:, :, :-1]  # Fill positions t>=2 with y values from t-1
            past_y_flat = past_y.view(-1, self.n_timepoints, 1)

            batch_size, max_meas, input_dim = x.shape
            # Generate causal mask
            causal_mask = self.generate_causal_mask(self.n_timepoints).to(x.device)

            # ---------------------------- VAE ----------------------------
            x_flat = x.view(-1, input_dim)  # (batch_size * max_measurements, input_dim)
            mu, logvar = self.encode(x_flat)
            z_hat = self.reparameterize(mu, logvar)
            x_hat_flat = self.decode(z_hat)  # Shape: [batch_size, input_dim]
            x_hat = x_hat_flat.view(batch_size, max_meas, input_dim)  # Reshape back

            # --------------------- Expand over time dimension ---------------------
            h = x_hat_flat.unsqueeze(1).repeat(1, self.n_timepoints, 1)  # [batch_size*M, T, input_dim]

            # --------------------- Add the lagged outcome y ----------------------
            h_with_lag_y = torch.cat([h, past_y_flat], dim=-1)

            # --------------- Projection to transformer input dimension -----------
            h_in = self.projection_to_transformer(h_with_lag_y)
            
            # --------------------- Time positional embedding ---------------------
            h_time = self.pos_encoder(h_in)

            attn_weights = self.transformer_module.cross_attn.get_attention_weights(
                h_time,
                h_time,
                attn_mask=causal_mask
            )
        
        return attn_weights


def loss_components(x, y, x_hat, y_hat, mu, logvar):
    
    # Reconstruction loss (MSE)
    reconstruction_loss = nn.functional.mse_loss(x_hat, x, reduction='none')
    # KL divergence
    KLD = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    # label prediction loss
    prediction_loss = nn.functional.mse_loss(y_hat, y, reduction='none')

    return reconstruction_loss, KLD, prediction_loss


# 4. Training Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

latent_dim = k * 2
# for the number of heads
transformer_input_dim = 256
transformer_dim_feedforward = transformer_input_dim * 4

model = TimeAttentionVAE(
    input_dim=p,
    n_timepoints=n_timepoints,
    n_measurements=n_measurements,
    vae_latent_dim=latent_dim,
    vae_input_to_latent_dim=64,
    max_len_position_enc=10,
    transformer_input_dim=transformer_input_dim,
    transformer_dim_feedforward=transformer_dim_feedforward,
    nheads=4,
    dropout=0.1,
    dropout_attention=0.1,
    prediction_weight=1.0
).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print(model)

#########################################################
x = next(iter(train_dataloader))[0]
batch_size, max_meas, input_dim = x.shape

past_y = torch.zeros(next(iter(train_dataloader))[1].shape)
past_y = past_y[..., None]
past_y[:, :, 1:, 0] = next(iter(train_dataloader))[1][:, :, :-1]  # Fill positions t>=2 with y values from t-1

# Generate causal mask
causal_mask = model.generate_causal_mask(model.n_timepoints).to(x.device)

# ------- VAE -------
x_flat = x.view(-1, input_dim)  # (batch_size * max_measurements, input_dim)
mu, logvar = model.encode(x_flat)
z_hat = model.reparameterize(mu, logvar)
x_hat_flat = model.decode(z_hat)  # Shape: [batch_size, input_dim]
x_hat = x_hat_flat.view(batch_size, max_meas, input_dim)  # Reshape back
x_hat.shape

# --- Time-Aware Prediction ---
h = x_hat_flat.unsqueeze(1).repeat(1, model.n_timepoints, 1)  # [batch_size*M, T, input_dim]
h.shape

# --------------------- Add the lagged outcome y ----------------------
past_y.shape
past_y_flat = past_y.view(-1, model.n_timepoints, 1)
past_y_flat.shape
h_with_lag_y = torch.cat([h, past_y_flat], dim=-1)
h_with_lag_y.shape

# --------------- Projection to transformer input dimension -----------
h_in = model.projection_to_transformer(h_with_lag_y)
h_in.shape

# --------------------- Time positional embedding ---------------------
h_time = model.pos_encoder(h_in)
h_time.shape

# --------- Transformer ---------
# This custom Transformer module expects input with shape: [batch_size, seq_len, input_dim]
h_out = model.transformer_module(h_time, attn_mask=causal_mask)
h_out.shape
# check attn weights
attn_weights = model.transformer_module.cross_attn.get_attention_weights(h_time, h_time, attn_mask=causal_mask)
attn_weights.shape
model.get_attention_weights(x)

# Predict outcomes
y_hat_flat = model.fc_out(model.dropout(h_out)).squeeze(-1)  # [batch_size*M, T]
y_hat = y_hat_flat.view(batch_size, max_meas, model.n_timepoints)
y_hat.shape

##########################################################

# 5. Training Loop
num_epochs = 200
# c_annealer = utils.CyclicAnnealer(cycle_length=num_epochs / 2, min_beta=0.0, max_beta=1.0, mode='cosine')
# plt.plot([c_annealer.get_beta(ii) for ii in range(1,num_epochs)])
# plt.show()

trainer = training.Training(train_dataloader, val_dataloader)

trainer.training_loop(model, optimizer, num_epochs)

plt.plot(trainer.losses["train"], label="train")
plt.plot(trainer.losses["val"], label="val")
plt.vlines(np.argmin(trainer.losses["val"]), 0, max(trainer.losses["val"]), color="red")
plt.vlines(np.argmin(trainer.losses["train"]), 0, max(trainer.losses["train"]), color="blue")
plt.hlines(np.min(trainer.losses["val"]), 0, len(trainer.losses["val"]), color="red", linestyles="--")
plt.hlines(np.min(trainer.losses["train"]), 0, len(trainer.losses["val"]), color="blue", linestyles="--")
plt.legend()
plt.show()

# optional
trainer.best_val_loss / len(val_dataloader.dataset)
model.load_state_dict(trainer.best_model.state_dict())

# check attention weights
attn_weights = model.get_attention_weights(data_train)
attn_weights.shape
plot_attention_weights(attn_weights, observation=10, layer_name="Attention")

# 6. Latent Space Extraction
model.eval()
with torch.no_grad():
    mu = model(tensor_data_test)[2]
    Z_hat = mu.cpu().numpy()
Z_hat.shape

plt.imshow(np.corrcoef(Z_hat, rowvar=False), cmap='jet', interpolation=None)
plt.colorbar()
plt.show()

# Features space reconstruction
model.eval()
with torch.no_grad():
    X_hat = model(tensor_data_test)[0].cpu().numpy()
X_hat.shape
np.corrcoef(data_test[0][:, 0, :], X_hat[:, 0, :], rowvar=False)

# plot
plt.scatter(X_hat[:, 0, 0], data_test[0][:, 0, 0])
plt.show()
plt.scatter(X_hat[:, 0, p-1], data_test[0][:, 0, p-1])
plt.show()

# LOSS components
with torch.no_grad():
    test_pred = model(tensor_data_test)
loss_x, loss_kl, loss_y = loss_components(
    tensor_data_test[0], tensor_data_test[1],
    x_hat=test_pred[0], y_hat=test_pred[1], mu=test_pred[2], logvar=test_pred[3]
)

print(loss_kl.mean())
print(loss_x.mean())
print(loss_y.mean())

plt.plot(loss_y.squeeze().numpy()[:, 0, :], linestyle="", marker="o")
plt.show()

# loss X
plt.plot(loss_x[:, 0].squeeze().numpy(), linestyle="", marker="o")
plt.show()

# average loss over dimensions
plt.plot(loss_x.mean(axis=1).squeeze().numpy(), linestyle="", marker="o")
plt.show()
# average loss over observations
plt.plot(loss_x.mean(axis=0).squeeze().numpy(), linestyle="", marker="o")
plt.show()


# Saliency map for feature importance
# weights from input to latent
x_train = tensor_data_train
x_train.requires_grad_(True)
x_hat, y_hat, _, _ = model(tensor_data_train)

t_point = 0
measurement = 0
y_hat[:, measurement, t_point].sum().backward()  # Focus on t=2
saliency = x_train.grad.abs().mean(dim=0)[measurement, :]  # [input_dim]
saliency.shape

# Plot top features
top_k = 10
top_indices = saliency.argsort(descending=True)[:top_k]
plt.bar(range(top_k), saliency[top_indices])
plt.xticks(range(top_k), top_indices.numpy(), rotation=45)
plt.title(f"Top Features Influencing t={t_point}")
plt.show()


# Coefficients
from prettytable import PrettyTable
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

count_parameters(model)

# Outcome predictions
# Features space reconstruction
model.eval()
with torch.no_grad():
    y_test_hat = model(tensor_data_test)[1].cpu().numpy()

pearsonr(data_test[2], y_test_hat)[0]
np.sqrt(np.mean((y_test_hat - data_test[2])**2, axis=0))

# Get colors from Pastel1 colormap
colors = plt.cm.Pastel1.colors

observation = 0
# Create a simple plot using these colors
fig, ax = plt.subplots(figsize=(8, 6))
for i in range(data_test[2][observation].shape[0]):
    ax.plot(data_test[2][observation][i], label="true", color=colors[i], linewidth=2)
    ax.plot(y_test_hat[observation][i], label="pred", color=colors[i], linewidth=2, linestyle="dashed")
ax.legend()
plt.show()
