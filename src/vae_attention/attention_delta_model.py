# Delta-learning model
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

from utils import training_wrapper

from utils import data_loading_wrappers
from utils.model_output_details import count_parameters
from utils import plots
from utils import data_generation

from vae_attention.modules.transformer import TransformerEncoderLayerWithWeights
from vae_attention.modules.sinusoidal_position_encoder import SinusoidalPositionalEncoding
from vae_attention.modules.vae import VAE


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


# y0 (y at baseline) is actually an additional feature, because it is measured before any intervention
y.shape
y_baseline = y[:, :, 0:1]
y_baseline.shape
# the actual target is then y from t=1
y_target = y[:, :, 1:]
y_target.shape

# get tensors
X_tensor = torch.FloatTensor(X).to(torch.device("cpu"))
Z_tensor = torch.FloatTensor(Z).to(torch.device("cpu"))
y_target_tensor = torch.FloatTensor(y_target).to(torch.device("cpu"))
y_baseline_tensor = torch.FloatTensor(y_baseline).to(torch.device("cpu"))

# split data
data_split = data_loading_wrappers.DataSplit(X.shape[0], test_size=n_test, val_size=n_val)
print("train: ", len(data_split.train_index), 
    "val: ", len(data_split.val_index),
    "test: ", len(data_split.test_index)
)

data_train = data_split.get_train(X, y_baseline, Z, y_target)
data_test = data_split.get_test(X, y_baseline, Z, y_target)
data_val = data_split.get_val(X, y_baseline, Z, y_target)

tensor_data_train = data_split.get_train(X_tensor, y_baseline_tensor, y_target_tensor)
tensor_data_test = data_split.get_test(X_tensor, y_baseline_tensor, y_target_tensor)
tensor_data_val = data_split.get_val(X_tensor, y_baseline_tensor, y_target_tensor)

# make tensor data loaders
train_dataloader = data_loading_wrappers.make_data_loader(*tensor_data_train, batch_size=batch_size)
test_dataloader = data_loading_wrappers.make_data_loader(*tensor_data_test, batch_size=batch_size)
val_dataloader = data_loading_wrappers.make_data_loader(*tensor_data_val, batch_size=batch_size)

next(iter(train_dataloader))[0].shape  # X
next(iter(train_dataloader))[1].shape  # y_baseline
next(iter(train_dataloader))[2].shape  # y_target


# VAE + Explicit multi-head Attention layer
class DeltaTimeAttentionVAE(nn.Module):
    """
    Model for longitudinal data with repeated measurements over time and another dimension (for example multiple interventions).
    
    Args:
        input_dim (int): Total dimension of the model.
        n_timepoints (int): Time points.
        n_measurements (int): Number of repeated measurements.
        vae_latent_dim (int): VAE latent space dimension.
        nheads (int): Number of attention heads.
        dropout (float, optional): Dropout probability. Default: 0.0
        dropout_attention (float, optional): Dropout probability. Default: 0.0
        activation: Add activation to output FFN. Default: gelu
    """
    def __init__(
        self,
        input_dim,
        n_timepoints,
        n_measurements,
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
        
        self.beta = beta_vae
        self.reconstruction_weight = reconstruction_weight
        self.prediction_weight = prediction_weight
        self.n_timepoints = n_timepoints
        self.n_measurements = n_measurements
        self.nheads = nheads
        self.transformer_input_dim = transformer_input_dim
        self.input_dim = input_dim
        # variables updated at each iteration of the training
        self.batch_size = 0
        self.max_meas = 0

        # Generate causal mask
        self.causal_mask = self.generate_causal_mask(n_timepoints)

        # Dropout
        self.dropout = nn.Dropout(dropout)

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
        self.fc_out = nn.Linear(self.transformer_input_dim, 1)  # Predicts 1 value per timepoint
    
    def generate_causal_mask(self, T):
        return torch.triu(torch.ones(T, T) * float('-inf'), diagonal=1)

    def generate_measurement_mask(self, batch_size, M):
        return torch.ones([batch_size, M])

    def preprocess_input(self, batch):
        # x has shape (n x M x p)
        x = batch[0]
        self.batch_size, self.max_meas, _ = x.shape

        # y has shape (n x M x T), so take only the first time point
        y_baseline = batch[1]
        y_baseline_flat = y_baseline.view(-1, 1)
        
        return x, y_baseline_flat

    def make_transformer_input(self, x_hat_flat, y0_flat):

        # --------------------- Add the lagged outcome y ----------------------
        h = torch.cat([x_hat_flat, y0_flat], dim=-1)
        # --------------------- Expand over time dimension ---------------------
        h_exp = h.unsqueeze(1).repeat(1, self.n_timepoints, 1)  # [batch_size*M, T, input_dim]
        # --------------- Projection to transformer input dimension -----------
        h_in = self.projection_to_transformer(h_exp)
        # --------------------- Time positional embedding ---------------------
        h_time = self.pos_encoder(h_in)

        return h_time
    
    def outcome_prediction(self, h_out):
        y_hat_flat = self.fc_out(self.dropout(h_out)).squeeze(-1)  # [batch_size*M, T]
        y_hat = y_hat_flat.view(self.batch_size, self.max_meas, self.n_timepoints)

        return y_hat

    def forward(self, batch):
        # ------------------ process input batch ------------------
        x, y_baseline_flat = self.preprocess_input(batch)

        # ---------------------------- VAE ----------------------------
        x_flat = x.view(-1, self.input_dim)  # (batch_size * max_measurements, input_dim)
        x_hat_flat, mu, logvar = self.vae(x_flat)
        x_hat = x_hat_flat.view(self.batch_size, self.max_meas, self.input_dim)  # Reshape back
        
        # ------ concatenate with y0, positional encoding and projection ------
        h_time = self.make_transformer_input(x_hat_flat, y_baseline_flat)

        # ----------------------- Transformer ------------------------------
        # This custom Transformer module expects input with shape: [batch_size, seq_len, input_dim]
        h_out = self.transformer_module(h_time, attn_mask=self.causal_mask)

        # --------------------- Predict outcomes ---------------------
        y_hat = self.outcome_prediction(h_out)
        
        return x_hat, y_hat, mu, logvar

    def loss(self, m_out, batch):
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
            x: Input tensors, same shape as used in training
        
        Returns:
            attn_weights: Attention weights tensor [batch_size, nhead, seq_len, seq_len]
        """
        with torch.no_grad():
            # ------------------ process input batch ------------------
            x, y0_flat = self.preprocess_input(batch)

            # ---------------------------- VAE ----------------------------
            x_flat = x.view(-1, self.input_dim)  # (batch_size * max_measurements, input_dim)
            x_hat_flat, mu, logvar = self.vae(x_flat)
            x_hat = x_hat_flat.view(self.batch_size, self.max_meas, self.input_dim)  # Reshape back

            # ------ concatenate with y0, positional encoding and projection ------
            h_time = self.make_transformer_input(x_hat_flat, y0_flat)

            attn_weights = self.transformer_module.cross_attn.get_attention_weights(
                h_time,
                h_time,
                attn_mask=self.causal_mask
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

model = DeltaTimeAttentionVAE(
    input_dim=p,
    n_timepoints=n_timepoints-1,
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
# Coefficients
count_parameters(model)

#########################################################
# x has shape (n x M x p)
batch = next(iter(train_dataloader))
# ------------------ process input batch ------------------
x, y0_flat = model.preprocess_input(batch)

# ---------------------------- VAE ----------------------------
x_hat, x_hat_flat, mu, logvar = model.vae_module(x)

# ------ concatenate with y0, positional encoding and projection ------
h_time = model.make_transformer_input(x_hat_flat, y0_flat)

# ----------------------- Transformer ------------------------------
# This custom Transformer module expects input with shape: [batch_size, seq_len, input_dim]
h_out = model.transformer_module(h_time, attn_mask=model.causal_mask)

# --------------------- Predict outcomes ---------------------
y_hat = model.outcome_prediction(h_out)

##########################################################

# 5. Training Loop
num_epochs = 200

trainer = training_wrapper.Training(train_dataloader, val_dataloader)

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
attn_weights = model.get_attention_weights(tensor_data_train)
attn_weights.shape
plots.plot_attention_weights(attn_weights, observation=10, layer_name="Attention")

# 6. Latent Space Extraction
model.eval()
with torch.no_grad():
    mu = model(tensor_data_test)[2]
    Z_hat = mu.cpu().numpy()

plt.imshow(np.corrcoef(Z_hat, rowvar=False), cmap='jet', interpolation=None)
plt.colorbar()
plt.show()


# Features space reconstruction
model.eval()
with torch.no_grad():
    X_hat = model(tensor_data_test)[0].cpu().numpy()
X_hat.shape

# plot
plt.scatter(X_hat[:, 0, 0], data_test[0][:, 0, 0])
plt.show()
plt.scatter(X_hat[:, 0, p-1], data_test[0][:, 0, p-1])
plt.show()


# LOSS components
with torch.no_grad():
    test_pred = model(tensor_data_test)
loss_x, loss_kl, loss_y = loss_components(
    x=tensor_data_test[0],
    y=tensor_data_test[2],
    x_hat=test_pred[0],
    y_hat=test_pred[1],
    mu=test_pred[2],
    logvar=test_pred[3]
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


# VAE Latent space contributions from X
def from_input_to_latent(model, x):
    with torch.no_grad():
        mu, logvar = model.encode(x)

    return mu, logvar

x = tensor_data_train[0]
x = x.view(-1, model.input_dim)  # (batch_size * max_measurements, input_dim)
mu, logvar = from_input_to_latent(model, x)
mu.shape


# Using SHAP
model.eval()
# Define a prediction function for the outcome y
def shap_predict(x):
    x = torch.FloatTensor(x)
    mu, _ = from_input_to_latent(model, x)
    return mu.numpy()

x = tensor_data_train[0][0:50]
x = x.view(-1, model.input_dim).numpy()  # (batch_size * max_measurements, input_dim)
x.shape
shap_predict(x).shape

# Create KernelExplainer
explainer = shap.KernelExplainer(shap_predict, x)  # Using 100 samples as background

samples_to_explain = tensor_data_test[0][0:50].view(-1, model.input_dim).numpy()
shap_values = explainer.shap_values(samples_to_explain)
shap_values.shape # (n x p x k)

# Plot feature importance
feature_names = [f'Feature {i}' for i in range(p)]

latent_axis = 0
shap.summary_plot(shap_values[:, :, latent_axis], samples_to_explain, show = True)

shap.plots.beeswarm(shap.Explanation(
    values=shap_values[:,:, latent_axis],
    base_values=explainer.expected_value[latent_axis], 
    data=samples_to_explain, 
    feature_names=feature_names
    ),
    max_display=6
)

feature = np.argmax(np.abs(shap_values[:,:, latent_axis]).sum(axis=0))
feature

fig = plt.figure()
plt.violinplot(shap_values[:, feature, :])
plt.xlabel("Latent Space Dimensions")
# set style for the axes
labels = range(latent_dim)
fig.axes[0].set_xticks(np.arange(1, len(labels) + 1), labels=labels)
fig.show()


shap.plots.heatmap(shap.Explanation(
    values=shap_values[:,:, latent_axis], 
    base_values=explainer.expected_value[latent_axis], 
    data=samples_to_explain, 
    feature_names=feature_names
    )
)

sample_ind = 0
shap.force_plot(
    explainer.expected_value[latent_axis],
    shap_values[sample_ind, :, latent_axis],
    samples_to_explain[sample_ind, :], 
    feature_names=[f'Feature {i}' for i in range(p)],
    matplotlib=True
)


# Looking for clusters using HDBSCAN
from sklearn.cluster import HDBSCAN
mean_shap_per_feature = shap_values.mean(axis=0)

hdb = HDBSCAN(min_cluster_size=5)
hdb.fit(mean_shap_per_feature)
np.unique(hdb.labels_).tolist()

# Visualize the clusters
plt.figure(figsize=(10, 6))
plt.scatter(
    mean_shap_per_feature[:, 1], mean_shap_per_feature[:, 2],
    c=hdb.labels_,
    cmap='viridis', s=50, alpha=0.7, edgecolors='k')
plt.colorbar()
plt.title('HDBSCAN Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()


# Latent space analysis through the decoder
# and
# Latent space perturbation
# Start from a baseline input for ONE observation
baseline_input = tensor_data_train[0][0:1]
batch_size, max_meas, _ = baseline_input.shape
x_flat = baseline_input.view(-1, model.input_dim)  # (batch_size * max_measurements, input_dim)
x_flat.shape

with torch.no_grad():
    mu, logvar = model.encode(x_flat)
    z_hat = model.reparameterize(mu, logvar)
    x_hat_flat = model.decode(z_hat)
    x_hat = x_hat_flat.view(batch_size, max_meas, model.input_dim)

# Choose a latent dimension to perturb and a perturbation amount
perturbation_range = np.linspace(-2, 2, 5) # perturb from -2 to 2 std dev
list_sensitivities = []

for latent_dim_to_perturb in range(latent_dim):
    reconstruction_changes = [] # List to store change per feature

    for eps in perturbation_range:
        z_perturbed = z_hat.clone()
        z_perturbed[0, latent_dim_to_perturb] += eps
        with torch.no_grad():
            pert_x_hat_flat = model.decode(z_perturbed)
            pert_x_hat = pert_x_hat_flat.view(batch_size, max_meas, model.input_dim)
            # take difference
            delta = pert_x_hat_flat - x_hat_flat
            reconstruction_changes.append(delta.numpy())

    # reconstruction_changes is a matrix [n_perturbations, p_features]
    reconstruction_changes = np.array(reconstruction_changes)
    reconstruction_changes.shape

    # For each feature, calculate its sensitivity to the latent dimension
    # (e.g., variance across perturbations or max absolute change)
    feature_sensitivity = np.var(reconstruction_changes, axis=0)
    feature_sensitivity.shape
    np.round(feature_sensitivity, 2)
    average_feature_sensitivity = feature_sensitivity.mean(axis=0)
    # feature_sensitivity = np.max(np.abs(reconstruction_changes), axis=0)
    list_sensitivities.append(average_feature_sensitivity)

top_sensitive_features = np.argsort(average_feature_sensitivity)[::-1][:10]
print(f"Features most sensitive to latent dim {latent_dim_to_perturb}: \n {top_sensitive_features}")

for latent_dim_to_perturb in range(latent_dim):
    plt.scatter(
        range(list_sensitivities[latent_dim_to_perturb].shape[0]),
        list_sensitivities[latent_dim_to_perturb],
        s=12,
        marker=latent_dim_to_perturb,
        label=f"Latent Dimension {latent_dim_to_perturb}"
    )
plt.legend()
plt.show()
