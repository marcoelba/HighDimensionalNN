# latent space recovery
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import numpy as np
from scipy.linalg import toeplitz
from scipy.stats import pearsonr
from scipy.linalg import orthogonal_procrustes

from sklearn.preprocessing import StandardScaler
from prettytable import PrettyTable

import matplotlib.pyplot as plt
import shap

import os
os.chdir("./src")

from vae_regression import data_generation
from vae_regression import training

from model_utils import utils


# generate data assuming an underlying latent space of dimension k
k = 5
n_train = 100
n_test = 200
n_val = 200
n = n_train + n_val + n_test
p = 500
p1 = 30
p0 = p - p1
n_timepoints = 4
n_measurements = 5

# custom W
np.random.seed(323425)
W = np.random.choice(
    [-1.5, -1, -0.8, -0.5, 1.5, 1, 0.8, 0.5],
    size=(k, p)
)
first_half = range(0, int(p/2))
second_half = range(int(p/2), p)
# # block structure
# W[0, first_half] = 0.0
# W[1, first_half] = 0.0
# W[3, second_half] = 0.0
# W[4, second_half] = 0.0
# # first p0 features do NOT have any effect
W[:, 0:p0] = 0

beta = np.array([-1, 1, -1, 1, -1])
beta = beta[..., None] * np.ones([k, n_timepoints])
beta[0, 1:] = [-2., -2., -1.]
beta[1, 1:] = [2., 3., 1.]
beta[2, 1:] = [0, 0, 0]

beta_time = np.array([0, 1, 2, 0, -1])

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
train_dataloader = utils.make_data_loader(*tensor_data_train, batch_size=32)
test_dataloader = utils.make_data_loader(*tensor_data_test, batch_size=32)
val_dataloader = utils.make_data_loader(*tensor_data_val, batch_size=32)

next(iter(train_dataloader))[0].shape  # X
next(iter(train_dataloader))[1].shape  # y

# x = next(iter(train_dataloader))[0]
# x = x[0:3, :, :]
# x.shape
# x[0, 1, :] = np.nan
# x[1, 2:4, :] = np.nan
# 1 - np.isnan(x[:, :, 0])

# x = torch.tensor(x)
# torch.logical_not(torch.isnan(x[:, :, 0])) * x[:, :, 0]

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


# 4. Training Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

latent_dim = k * 2
# for the number of heads
(p + 8) / 4
(p + 8) * 2
p / 2

model = TimeAwareRegVAE(
    input_dim=p,
    latent_dim=latent_dim,
    n_timepoints=n_timepoints,
    n_measurements=n_measurements,
    input_to_latent_dim=256,
    transformer_dim_feedforward=1016,
    nhead=4,
    time_emb_dim=8,
    dropout_sigma=0.0,
    beta_vae=1.0,
    reconstruction_weight=1.0,
    prediction_weight=1.0
).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)



#########################################################
x = next(iter(train_dataloader))[0]
batch_size, max_meas, input_dim = x.shape

# --- VAE Forward Pass ---
# Generate causal mask
causal_mask = model.generate_causal_mask(model.n_timepoints).to(x.device)

x_flat = x.view(-1, input_dim)  # (batch_size * max_measurements, input_dim)
x_flat.shape
mu, logvar = model.encode(x_flat)
z_hat = model.reparameterize(mu, logvar)
z_hat.shape
x_hat_flat = model.decode(z_hat)  # Shape: [batch_size, input_dim]
x_hat_flat.shape
x_hat = x_hat_flat.view(batch_size, max_meas, input_dim)  # Reshape back
x_hat.shape

# --- Time-Aware Prediction ---
# Project latent features
# h = model.latent_proj(x_hat_flat)  # Shape: [batch_size*M, 32]
# h.shape
# h = h.unsqueeze(1).repeat(1, model.n_timepoints, 1)  # [batch_size*M, T, 32]
# h.shape

h = x_hat_flat.unsqueeze(1).repeat(1, model.n_timepoints, 1)  # [batch_size*M, T, 32]
h.shape

# Get time embeddings (for all timepoints)
time_ids = torch.arange(model.n_timepoints, device=x.device)  # [0, 1, ..., T-1]
time_embs = model.time_embedding(time_ids)  # [T, time_emb_dim]
time_embs.shape
time_embs = time_embs.unsqueeze(0).repeat(h.shape[0], 1, 1)  # [1, T, time_emb_dim]

# Combine latent features and time embeddings
h_time = torch.cat([h, time_embs], dim=-1)  # [batch_size*M, T, 32 + time_emb_dim]
h_time.shape

# Process temporally
# Transformer expects [T, batch_size, features]
h_time = h_time.transpose(0, 1)  # [T, batch_size*M, ...]
h_out = model.transformer(h_time, mask=causal_mask)  # [T, batch_size, ...]
h_out = h_out.transpose(0, 1)    # [batch_size*M, T, ...]
h_out.shape

# Predict outcomes
y_hat_flat = model.fc_out(model.dropout(h_out)).squeeze(-1)  # [batch_size*M, T]
y_hat_flat.shape
y_hat = y_hat_flat.view(batch_size, max_meas, model.n_timepoints)
y_hat.shape

##########################################################


# 5. Training Loop
num_epochs = 100
# c_annealer = utils.CyclicAnnealer(cycle_length=num_epochs / 2, min_beta=0.0, max_beta=1.0, mode='cosine')
# plt.plot([c_annealer.get_beta(ii) for ii in range(1,num_epochs)])
# plt.show()

trainer = training.Training(train_dataloader, val_dataloader)

trainer.training_loop(model, optimizer, num_epochs)

plt.plot(trainer.losses["train"], label="train")
plt.plot(trainer.losses["val"], label="val")
plt.vlines(np.argmin(trainer.losses["val"]), 0, max(trainer.losses["val"]), color="red")
plt.hlines(np.min(trainer.losses["val"]), 0, num_epochs, color="red", linestyles="--")
plt.legend()
plt.show()

trainer.best_val_loss / len(val_dataloader.dataset)
model.load_state_dict(trainer.best_model.state_dict())

# 6. Latent Space Extraction
model.eval()
with torch.no_grad():
    mu = model(tensor_data_test[0].to(device))[2]
    Z_hat = mu.cpu().numpy()

if latent_dim == k:
    # 7. Evaluation (Compare with original Z)
    # Procrustes alignment
    R, _ = orthogonal_procrustes(Z_hat[:,1,:], data_test[1])
    Z_aligned = Z_hat[:,1,:] @ R

utils.print_correlations(data_test[1], Z_hat[:,1,:])
utils.print_correlations(data_test[1], Z_aligned)


plt.imshow(np.corrcoef(Z_hat, rowvar=False), cmap='jet', interpolation=None)
plt.colorbar()
plt.show()


# Features space reconstruction
model.eval()
with torch.no_grad():
    X_hat = model(tensor_data_test[0])[0].cpu().numpy()
X_hat.shape
np.corrcoef(data_test[0][:, 0, :], X_hat[:, 0, :], rowvar=False)

# plot
plt.scatter(X_hat[:, 0, 0], data_test[0][:, 0, 0])
plt.show()
plt.scatter(X_hat[:, 0, p-1], data_test[0][:, 0, p-1])
plt.show()

# LOSS components
with torch.no_grad():
    test_pred = model(tensor_data_test[0])
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
x_train = tensor_data_train[0]
x_train.requires_grad_(True)
x_hat, y_hat, _, _ = model(x_train)

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

# Outcome predictions
# Features space reconstruction
model.eval()

with torch.no_grad():
    y_test_hat = model(tensor_data_test[0])[1].cpu().numpy()

pearsonr(data_test[2], y_test_hat)[0]
np.sqrt(np.mean((y_test_hat - data_test[2])**2, axis=0))

plt.plot(data_test[2][0], label="true")
plt.plot(y_test_hat[0], label="pred")
plt.legend()
plt.show()

# Coefficients
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

# --------------- SHAP explanations ---------------
model.eval()

# Define a prediction function for the outcome y
def predict(x):
    x = torch.FloatTensor(x)
    with torch.no_grad():
        preds = model.predict(x)
    return preds.numpy()

# ---------- reshape data to long format for SHAP -------
test_size, max_meas, input_dim = data_train[0].shape
x_flat = tensor_data_train[0][0:30].view(-1, input_dim)  # (batch_size * max_measurements, input_dim)
x_flat = x_flat.detach().numpy()
x_flat.shape
predict(x_flat).shape

# Create KernelExplainer
explainer = shap.KernelExplainer(predict, x_flat)  # Using 100 samples as background
# Deep explainer for NN
# model.eval()
# explainer = shap.DeepExplainer((model, model.fc1), X_train_tensor)

samples_to_explain = tensor_data_test[0][0:25]
samples_to_explain = samples_to_explain.view(-1, input_dim)  # (batch_size * max_measurements, input_dim)
samples_to_explain = samples_to_explain.detach().numpy()
samples_to_explain.shape

shap_values = explainer.shap_values(samples_to_explain)
shap_values.shape

# Plot feature importance
feature_names = [f'Feature {i}' for i in range(p)]

time_point = 0
shap.summary_plot(shap_values[:, :, time_point], samples_to_explain, show = True)

shap.plots.beeswarm(shap.Explanation(
    values=shap_values[:,:,time_point],
    base_values=explainer.expected_value[time_point], 
    data=samples_to_explain, 
    feature_names=feature_names
    ),
    max_display=6
)

feature = np.argmax(np.abs(shap_values[:,:,time_point]).sum(axis=0))

fig = plt.figure()
plt.violinplot(shap_values[:, feature, :])
plt.xlabel("Time")
# set style for the axes
labels = [1, 2, 3, 4]
fig.axes[0].set_xticks(np.arange(1, len(labels) + 1), labels=labels)
fig.show()


shap.plots.heatmap(shap.Explanation(
    values=shap_values[:,:,time_point], 
    base_values=explainer.expected_value[time_point], 
    data=samples_to_explain, 
    feature_names=feature_names
    )
)

sample_ind = 0
shap.force_plot(explainer.expected_value[time_point], shap_values[sample_ind, :, time_point], samples_to_explain[sample_ind, :], 
    feature_names=[f'Feature {i}' for i in range(p)], matplotlib=True
)


# Latent space visualisation
model.eval()
latents, labels = [], []
with torch.no_grad():
    for x, y in test_dataloader:
        mu, _ = model.encode(x)
        latents.append(mu)
        labels.append(y)
latents = torch.cat(latents).numpy()
labels = torch.cat(labels).numpy()
latents.shape
labels.shape

plt.scatter(latents[:, 0], latents[:, 3], c=labels, alpha=0.9)
plt.colorbar()
plt.show()


# Generate samples from prior
with torch.no_grad():
    z = torch.randn(n_test, latent_dim)  # Sample from N(0,1)
    generated = model.decode(z).cpu().numpy()

# Visualize generated samples
dim = 15
plt.hist(generated[:, 0, dim], label="gen")
plt.hist(data_test[0][:, 0, dim], label="real", alpha=0.5)
plt.legend()
plt.show()
