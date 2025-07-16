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
from model_utils.models import GaussianDropout


# generate data assuming an underlying latent space of dimension k
k = 5
n_train = 100
n_test = 200
n_val = 200
n = n_train + n_val + n_test
p = 20
n_timepoints = 4

# custom W
W = np.random.normal(size=(k, p))
first_half = range(0, int(p/2))
second_half = range(int(p/2), p)
# block structure
W[0, first_half] = 0.0
W[1, first_half] = 0.0
W[3, second_half] = 0.0
W[4, second_half] = 0.0
# first 10 features do NOT have any effect
W[:, 0:450] = 0

beta = np.array([-1, 1, -1, 1, -1])
beta_time = np.array([0, 1, 2, 0, -1])

y, X, Z, beta = data_generation.longitudinal_data_generation(
    n, k, p, n_timepoints,
    noise_scale = 0.5,
    W=W,
    beta=beta,
    beta_time=beta_time
)

plt.plot(y[0:10, :].transpose())
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


# VAE + regression
class TimeAwareRegVAE(nn.Module):
    def __init__(
        self,
        input_dim,          # Dimension of fixed input X (e.g., number of genes)
        latent_dim,         # Dimension of latent space Z
        n_timepoints=5,     # Number of timepoints (T)
        time_emb_dim=8,     # Dimension of time embeddings
        dropout_sigma=0.1,
        beta_vae=1.0
    ):
        super(TimeAwareRegVAE, self).__init__()
        
        self.beta = beta_vae
        self.n_timepoints = n_timepoints

        # --- VAE (unchanged) ---
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_var = nn.Linear(32, latent_dim)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

        # --- Time Embeddings ---
        self.time_embedding = nn.Embedding(n_timepoints, time_emb_dim)

        # --- Latent-to-Outcome Mapping ---
        # Project latent features to a space compatible with time embeddings
        self.latent_proj = nn.Linear(input_dim, 32)  # Projects x_hat

        # --- Time-Aware Prediction Head ---
        # Lightweight Transformer
        encoder_layer = TransformerEncoderLayer(
            d_model=32 + time_emb_dim,  # Input dim (latent + time)
            nhead=4,                    # Number of attention heads
            dim_feedforward=64,
            dropout=dropout_sigma,
            activation="gelu"
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=1)
        self.fc_out = nn.Linear(32 + time_emb_dim, 1)  # Predicts 1 value per timepoint

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
    
    def forward(self, x):
        # --- VAE Forward Pass ---
        mu, logvar = self.encode(x)
        z_hat = self.reparameterize(mu, logvar)
        x_hat = self.decode(z_hat)  # Shape: [batch_size, input_dim]

        # --- Time-Aware Prediction ---
        # Project latent features
        h = self.latent_proj(x_hat)  # Shape: [batch_size, 32]
        h = h.unsqueeze(1).repeat(1, self.n_timepoints, 1)  # [batch_size, T, 32]

        # Get time embeddings (for all timepoints)
        time_ids = torch.arange(self.n_timepoints, device=x.device)  # [0, 1, ..., T-1]
        time_embs = self.time_embedding(time_ids)  # [T, time_emb_dim]
        time_embs = time_embs.unsqueeze(0).repeat(x.size(0), 1, 1)  # [batch_size, T, time_emb_dim]

        # Combine latent features and time embeddings
        h_time = torch.cat([h, time_embs], dim=-1)  # [batch_size, T, 32 + time_emb_dim]

        # Process temporally
        # Transformer expects [T, batch_size, features]
        h_time = h_time.transpose(0, 1)  # [T, batch_size, ...]
        h_out = self.transformer(h_time)  # [T, batch_size, ...]
        h_out = h_out.transpose(0, 1)    # [batch_size, T, ...]

        # Predict outcomes
        y_hat = self.fc_out(self.dropout(h_out)).squeeze(-1)  # [batch_size, T]

        return x_hat, y_hat, mu, logvar

    def predict(self, x):
        # --- VAE Forward Pass ---
        mu, logvar = self.encode(x)
        z_hat = self.reparameterize(mu, logvar)
        x_hat = self.decode(z_hat)  # Shape: [batch_size, input_dim]

        # --- Time-Aware Prediction ---
        # Project latent features
        h = self.latent_proj(x_hat)  # Shape: [batch_size, 32]
        h = h.unsqueeze(1).repeat(1, self.n_timepoints, 1)  # [batch_size, T, 32]

        # Get time embeddings (for all timepoints)
        time_ids = torch.arange(self.n_timepoints, device=x.device)  # [0, 1, ..., T-1]
        time_embs = self.time_embedding(time_ids)  # [T, time_emb_dim]
        time_embs = time_embs.unsqueeze(0).repeat(x.size(0), 1, 1)  # [batch_size, T, time_emb_dim]

        # Combine latent features and time embeddings
        h_time = torch.cat([h, time_embs], dim=-1)  # [batch_size, T, 32 + time_emb_dim]

        # Process temporally
        # Transformer expects [T, batch_size, features]
        h_time = h_time.transpose(0, 1)  # [T, batch_size, ...]
        h_out = self.transformer(h_time)  # [T, batch_size, ...]
        h_out = h_out.transpose(0, 1)    # [batch_size, T, ...]

        # Predict outcomes
        y_hat = self.fc_out(self.dropout(h_out)).squeeze(-1)  # [batch_size, T]

        return y_hat

    def loss(self, m_out, x, y):
        # Reconstruction loss (MSE)
        BCE = nn.functional.mse_loss(m_out[0], x, reduction='sum')
        # KL divergence
        KLD = -0.5 * torch.sum(1 + m_out[3] - m_out[2].pow(2) - m_out[3].exp())
        # label prediction loss
        PredMSE = nn.functional.mse_loss(m_out[1], y, reduction='sum')

        return BCE + self.beta * KLD + PredMSE


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

latent_dim = k
model = TimeAwareRegVAE(
    input_dim=p,
    latent_dim=latent_dim,
    n_timepoints=n_timepoints,
    time_emb_dim=8,
    dropout_sigma=0.2,
    beta_vae=1.0
).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 5. Training Loop
num_epochs = 500
c_annealer = utils.CyclicAnnealer(cycle_length=num_epochs / 2, min_beta=0.0, max_beta=1.0, mode='cosine')
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
    mu, _ = model.encode(tensor_data_test[0].to(device))
    Z_hat = mu.cpu().numpy()

if latent_dim == k:
    # 7. Evaluation (Compare with original Z)
    # Procrustes alignment
    R, _ = orthogonal_procrustes(Z_hat, data_test[1])
    Z_aligned = Z_hat @ R

utils.print_correlations(data_test[1], Z_hat)
utils.print_correlations(data_test[1], Z_aligned)


plt.imshow(np.corrcoef(Z_hat, rowvar=False), cmap='jet', interpolation=None)
plt.colorbar()
plt.show()


# Features space reconstruction
model.eval()
with torch.no_grad():
    X_hat = model(tensor_data_test[0])[0].cpu().numpy()

utils.print_correlations(data_test[0], X_hat)

# plot
plt.scatter(X_hat[:, 0], data_test[0][:, 0])
plt.show()
plt.scatter(X_hat[:, 10], data_test[0][:, 10])
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

plt.plot(loss_y.squeeze().numpy(), linestyle="", marker="o")
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


# loss for just one observation
new_x = [
    torch.Tensor(np.random.normal(scale=1., size=(1, p))),
    torch.Tensor([[1.]])
]
with torch.no_grad():
    test_pred = model(new_x[0])
loss_x, loss_kl, loss_y = loss_components(
    new_x[0], new_x[1],
    x_hat=test_pred[0], y_hat=test_pred[1], mu=test_pred[2], logvar=test_pred[3]
)
loss_x.mean()
loss_y.mean()

# Saliency map for feature importance
# weights from input to latent
x_train = tensor_data_train[0]
x_train.requires_grad_(True)
x_hat, y_hat, _, _ = model(x_train)

t_point = 0
y_hat[:, t_point].sum().backward()  # Focus on t=2
saliency = x_train.grad.abs().mean(dim=0)  # [input_dim]

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
np.sqrt(np.mean((y_test_hat - data_test[2])**2))

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
    return preds.numpy().flatten()

# Create KernelExplainer
explainer = shap.KernelExplainer(predict, data_train[0])  # Using 100 samples as background
# Deep explainer for NN
# model.eval()
# explainer = shap.DeepExplainer((model, model.fc1), X_train_tensor)

samples_to_explain = data_test[0]
samples_to_explain.shape

shap_values = explainer.shap_values(samples_to_explain)
shap_values.shape

# Plot feature importance
feature_names = [f'Feature {i}' for i in range(p)]

shap.summary_plot(shap_values, samples_to_explain, show = True)
shap.summary_plot(shap_values, samples_to_explain, plot_type="violin", show = True)

shap.plots.heatmap(shap.Explanation(
    values=shap_values, 
    base_values=explainer.expected_value, 
    data=samples_to_explain, 
    feature_names=feature_names
    )
)

sample_ind = 0
shap.force_plot(explainer.expected_value, shap_values[sample_ind, :], samples_to_explain[sample_ind, :], 
    feature_names=[f'Feature {i}' for i in range(p)], matplotlib=True
)

# Takes into account features correlation
import sage

imputer = sage.MarginalImputer(predict, data_train[0][0:50, :])  # Background data
estimator = sage.KernelEstimator(imputer, 'mse')
sage_values = estimator(data_train[0], data_train[2].flatten())
sage_values.plot(feature_names, max_features=20)
plt.show()


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

plt.scatter(latents[:, 0], latents[:, 3], c=labels, alpha=0.9)
plt.colorbar()
plt.show()


# Generate samples from prior
with torch.no_grad():
    z = torch.randn(n_test, latent_dim)  # Sample from N(0,1)
    generated = model.decode(z).cpu().numpy()

# Visualize generated samples
dim = 15
plt.hist(generated[:, dim], label="gen")
plt.hist(data_test[0][:, dim], label="real", alpha=0.5)
plt.legend()
plt.show()


# ONLY linear model
from sklearn.linear_model import LassoCV

lasso = LassoCV(cv=5, random_state=0).fit(data_train[0], data_train[2].ravel())
lasso.score(data_train[0], data_train[2].ravel())
lasso.score(data_test[0], data_test[2].ravel())

plt.plot(lasso.coef_, color='green', marker='o', linestyle='')
plt.hlines(0, 0, p)
plt.show()

lasso_pred = lasso.predict(data_test[0])
np.corrcoef(data_test[2].ravel(), lasso_pred, rowvar=False)**2

np.sqrt(np.mean((lasso_pred - data_test[2].ravel())**2))
