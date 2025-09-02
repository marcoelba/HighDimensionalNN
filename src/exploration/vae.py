# latent space recovery
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.parametrizations import orthogonal

import numpy as np
from scipy.linalg import toeplitz
from scipy.stats import pearsonr
from scipy.linalg import orthogonal_procrustes

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import shap

import os
os.chdir("./src")

from model_utils import models, utils


# generate data assuming an underlying latent space of dimension k
k = 5
n = 500
n_test = 200
n_val = 200
p = 500

# set the covariance matrix
cov_matrix = np.random.rand(k, k) * 0.1
cov_matrix = cov_matrix @ cov_matrix.T  # positive semi-definite
cov_matrix
cov_matrix += 1 * np.diag(np.ones(k))
np.linalg.inv(cov_matrix)

# latent space
Z = np.random.multivariate_normal(mean=np.zeros(k), cov=cov_matrix, size=n)

# 2. Create transformation matrix W
W = np.random.normal(size=(k, p))

first_half = range(0, int(p/2))
second_half = range(int(p/2), p)
W[0, first_half] = 0.0
W[1, second_half] = 0.0
W[2, first_half] = 0.0

# 3. Compute X = ZW + noise
noise_scale = 0.5
X = Z @ W + np.random.normal(scale=noise_scale, size=(n, p))

# np.corrcoef(X, rowvar=False)
# plt.imshow(np.corrcoef(X, rowvar=False), cmap='jet', interpolation='nearest')
# plt.colorbar()
# plt.show()

# add outcome
y = Z @ np.random.choice([-1, 1], size=k) + np.random.normal(scale=noise_scale, size=n)
y = y[..., None]

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


data_train = data_split.get_train(X, y)
data_test = data_split.get_test(X, y)
data_val = data_split.get_val(X, y)

tensor_data_train = data_split.get_train(X_tensor, y_tensor)
tensor_data_test = data_split.get_test(X_tensor, y_tensor)
tensor_data_val = data_split.get_val(X_tensor, y_tensor)

# make tensor data loaders
train_dataloader = utils.make_data_loader(*tensor_data_train, batch_size=32)
test_dataloader = utils.make_data_loader(*tensor_data_test, batch_size=32)
val_dataloader = utils.make_data_loader(*tensor_data_test, batch_size=32)

# 4. Training Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

latent_dim = k
model = models.VAE(input_dim=p, latent_dim=latent_dim, beta=1).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 5. Training Loop
num_epochs = 300
c_annealer = utils.CyclicAnnealer(cycle_length=num_epochs / 4, min_beta=0.0, max_beta=1.0, mode='cosine')

trainer = utils.Training(train_dataloader, val_dataloader, device=device)
trainer.losses.keys()

trainer.training_loop(model, optimizer, num_epochs)

plt.plot(trainer.losses["train"], label="train")
plt.plot(trainer.losses["val"], label="val")
plt.legend()
plt.show()


# 6. Latent Space Extraction
model.eval()
with torch.no_grad():
    mu, logvar = model.encode(tensor_data_test[0].to(device))
    Z_test_hat = mu.cpu().numpy()
    var_hat = logvar.cpu().numpy()

# 7. Evaluation (Compare with original Z)
# Procrustes alignment
R, _ = orthogonal_procrustes(Z_test_hat, Z[data_split.test_index])
Z_aligned = Z_test_hat @ R

utils.print_correlations(Z[data_split.test_index], Z_aligned)
utils.print_correlations(Z[data_split.test_index], Z_test_hat)


# plot
plt.scatter(Z_aligned[:, 0], Z[data_split.test_index, 0])
plt.scatter(Z_test_hat[:, 0], Z[data_split.test_index, 0])
plt.show()

cor_Z = np.corrcoef(Z_test_hat, rowvar=False)
np.fill_diagonal(cor_Z, 0.0)
plt.imshow(cor_Z, cmap='jet', interpolation=None)
plt.colorbar()
plt.show()

var_hat.shape
for dim in range(latent_dim):
    plt.hist(var_hat[:, dim])
plt.show()
np.var(Z_test_hat, axis=0)


# Features space reconstruction
model.eval()
with torch.no_grad():
    X_test_hat = model(tensor_data_test[0].to(device))[0].cpu().numpy()

utils.print_correlations(tensor_data_test[0], X_test_hat)

# plot
plt.scatter(X_test_hat[:, 0], tensor_data_test[0][:, 0])
plt.show()


# Check the weights of the layers
plt.plot(model.encoder[0].weight.detach().numpy().flatten())
plt.show()

# ------------ Analyse the feature impact on the latent space with SHAP ------------
class DeterministicEncoder(torch.nn.Module):
    def __init__(self, vae_model):
        super().__init__()
        self.vae = vae_model
    
    def forward(self, x):
        # Return only the mean (μ) of the latent distribution
        mu, _ = self.vae.encode(x)  # Shape: (batch_size, latent_dim)
        return mu

# Wrap the VAE's encoder
deterministic_encoder = DeterministicEncoder(model)
deterministic_encoder.eval()

# Use a subset of training data to represent "typical" inputs
background = data_train[0]  # Shape: (100, input_dim)
background_tensor = torch.FloatTensor(background)

# Can also use a kernel explainer
explainer = shap.DeepExplainer(
    deterministic_encoder,
    background_tensor
)

# Select a test sample to explain
test_sample = data_test[0]  # Shape: (1, input_dim)
test_tensor = torch.FloatTensor(test_sample)
test_tensor.shape

# Get SHAP values for the latent mean (μ)
shap_values = explainer.shap_values(test_tensor)
shap_values.shape
feature_names = [f'F{i}' for i in range(p)]

# Plot for a specific latent dimension (e.g., dimension 0)
latent_dim = 0
shap.force_plot(
    explainer.expected_value[latent_dim],
    shap_values[:, :, latent_dim],
    test_sample,
    feature_names=feature_names,
    matplotlib=True
)

# Compute mean absolute SHAP values per latent-feature pair
mean_shap = np.mean(np.abs(shap_values), axis=0).transpose()  # Shape: (latent_dim, n_features)


plt.imshow(mean_shap[:, 230:270], cmap="jet")
plt.ylabel("Input Features")
plt.xlabel("Latent Dimensions")
plt.colorbar()
plt.title("Global SHAP Importance (Absolute Mean)")
plt.show()
