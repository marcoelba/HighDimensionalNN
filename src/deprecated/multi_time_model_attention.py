import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.parametrizations import orthogonal
import torch.nn.functional as F

import numpy as np
from scipy.linalg import toeplitz
from scipy.linalg import cholesky
from scipy.stats import pearsonr
from scipy.linalg import orthogonal_procrustes

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import shap
import pandas as pd

import os
os.chdir("./src")

from model_utils import models, utils


# generate data
n_patients = 500
n_timepoints = 5
n_measurements = 4
p = 10
noise_level = 0.1
feature_noise = 0.1      # Feature noise across measurements
missing_prob = 0.3       # Probability of a measurement being missing per patient

n_test = 100
n_val = 100

# 1. Genrate X baseline
X_base = np.random.normal(loc=0.0, scale=1.0, size=(n_patients, p)) # n x p

# 2. Extend X to (n_patients × p × n_measurements)
X = np.zeros((n_patients, p, n_measurements))
for m in range(n_measurements):
    X[:, :, m] = X_base + np.random.normal(0, feature_noise, size=(n_patients, p))

# 3. Generate binary mask for missing measurements
mask = np.random.binomial(1, 1 - missing_prob, size=(n_patients, n_measurements)) # n x M

# --- 4. Apply mask to X (set missing features to np.nan) ---
X_masked = X.copy()
for m in range(n_measurements):
    X_masked[mask[:, m] == 0, :, m] = np.nan


# 5. Simulate correlated outcomes (y) - error
corr_matrix = np.array([
    [1.0, 0.7, 0.3, 0.2],
    [0.7, 1.0, 0.5, 0.4],
    [0.3, 0.5, 1.0, 0.6],
    [0.2, 0.4, 0.6, 1.0]
])
L = cholesky(corr_matrix, lower=True)
correlated_noise = np.random.normal(0, 0.05, size=(n_patients, n_timepoints, n_measurements))
correlated_noise = np.einsum('ijk,lk->ijl', correlated_noise, L)
np.corrcoef(correlated_noise[:, 0, :], rowvar=False)


# Simulate Attention Weights, aka impact of features on time evolution
attn_weights = np.zeros((n_patients, n_timepoints, n_measurements))
for i in range(n_patients):
    for m in range(n_measurements):
        if X_base[i, 0] > 0:  # High severity
            attn_weights[i, :, m] = [0.1, 0.2, 0.3, 0.2, 0.2]
        else:                 # Low severity
            attn_weights[i, :, m] = [0.3, 0.4, 0.1, 0.1, 0.1]
attn_weights += correlated_noise
attn_weights = np.abs(attn_weights)
attn_weights /= attn_weights.sum(axis=1, keepdims=True)

# Simulate baseline feature effects
baseline = np.zeros((n_patients, n_measurements))
baseline[:, 0] = 1 + 2 * X_base[:, 0] + 1 * X_base[:, 1] + 1 * X_base[:, 2]
baseline[:, 1] = 1 + 2.1 * X_base[:, 0] + 1.1 * X_base[:, 1] + 1 * X_base[:, 2]
baseline[:, 2] = 1 + 1.9 * X_base[:, 0] + 1 * X_base[:, 1] + 1.1 * X_base[:, 2]
baseline[:, 3] = 1 + 1.8 * X_base[:, 0] + 0.9 * X_base[:, 1] + 0.9 * X_base[:, 2]

# and time effects
time = np.arange(n_timepoints)
time_effect = np.zeros((n_timepoints, n_measurements))

time_effect[:, 0] = 0.5 * time + 0.2 + np.sin(time)
time_effect[:, 1] = 0.3 * time + 0.1 + np.sin(time)
time_effect[:, 2] = 0.2 * time + 0.3 + np.sin(time)
time_effect[:, 3] = 0.4 * time + 0.1 + np.sin(time)

plt.plot(time_effect)
plt.show()

y = np.zeros((n_patients, n_timepoints, n_measurements)) # n x T x M
for i in range(n_patients):
    for m in range(n_measurements):
        y[i, :, m] = baseline[i, m] + time_effect[:, m]
        # y[i, :, m] += np.random.normal(0, 0.1, n_timepoints)
        y[i, :, m] *= attn_weights[i, :, m] * n_timepoints


plt.plot(y[0, :, :])
plt.show()

# Add correlated noise and apply mask to y
measurement_noise = np.random.multivariate_normal(
    mean=np.zeros(n_measurements),
    cov=corr_matrix * noise_level**2,
    size=(n_patients, n_timepoints)
)
y += measurement_noise
y_masked = y.copy()
for m in range(n_measurements):
    y_masked[mask[:, m] == 0, :, m] = np.nan


# --- Verify consistency ---
print("X_masked shape:", X_masked.shape)  # (n_patients, p, n_measurements)
print("y_masked shape:", y_masked.shape)  # (n_patients, T_timepoints, n_measurements)
print("Example missing measurements for patient 0:", np.where(np.isnan(X_masked[0, 0, :]))[0])
print("Example missing measurements for patient 1:", np.where(np.isnan(X_masked[4, 0, :]))[0])


# Create patient and measurement IDs
patient_ids = np.repeat(np.arange(n_patients), n_measurements)
measurement_ids = np.tile(np.arange(n_measurements), n_patients)

# Reshape X and y to LONG format for the measurement dimension
X_long = X_masked.transpose(0, 2, 1).reshape(-1, p)  # (n_patients*n_measurements, p)
y_long = y_masked.transpose(0, 2, 1).reshape(-1, n_timepoints)  # (n_patients*n_measurements, T)

print(X_long.shape)
print(y_long.shape)

# Mask for observed measurements (optional)
mask_long = ~np.isnan(X_long[:, 0])  # (n_patients*n_measurements,)
print(mask_long.shape)

data_split = utils.DataSplit(X.shape[0], test_size=n_test, val_size=n_val, unique_ids=patient_ids)
len(data_split.train_index)
len(data_split.test_index)
len(data_split.val_index)

data_train = data_split.get_train(X_long, y_long)
data_test = data_split.get_test(X_long, y_long)
data_val = data_split.get_val(X_long, y_long)


class MaskedTimeAttentionModel(nn.Module):
    def __init__(self, input_dim, time_steps, n_measurements, hidden_dim=32, embedding_dim=16):
        super().__init__()

        self.time_steps = time_steps
        self.n_measurements = n_measurements
        
        # Embeddings of categorical variables
        self.time_embed_layer = nn.Embedding(time_steps, embedding_dim)
        self.measurement_embed_layer = nn.Embedding(n_measurements, embedding_dim)
        
        # Feature mixing
        self.feature_mixer = nn.Linear(input_dim + embedding_dim + embedding_dim, hidden_dim)
        
        # Time-only attention
        self.time_attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, time_steps)
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, 1)
    
    def forward(self, data):
        """
        Args:
            x: Input tensor of shape (batch_size, p_features) where batch = n_patients*n_measurements
            measurement_ids: (batch,) indicating measurement ID
        """
        batch_size = data["X"].shape[0]

        # --- Time embeddings ---
        # here creating the range from 0 to T-1 to encode the dynamic nature of the time indices
        time = torch.arange(self.time_steps).repeat(batch_size, 1)  # (batch, T)
        time_embed = self.time_embed_layer(time)  # (batch, T, 16)
        
        # --- Measurement embeddings ---
        # while measurement is a static information
        measurement_embed = self.measurement_embed_layer(data["measurement_ids"]).unsqueeze(1)  # (batch, 1, 16)
        measurement_embed = measurement_embed.expand(-1, self.time_steps, -1)  # (batch, T, 16)

        # --- Feature mixing ---
        x_expanded = data["X"].unsqueeze(1).expand(-1, self.time_steps, -1)  # (batch, T, p)
        features = torch.cat([
            x_expanded,
            time_embed,
            measurement_embed
            ], dim=-1
        )
        features = features * data["X_mask"].unsqueeze(-1).unsqueeze(-1)  # Mask missing
        features = self.feature_mixer(features)  # (batch, T, hidden_dim)
        # features = torch.relu(features)

        # --- Time attention ---
        attn_scores = self.time_attention(data["X"])  # (batch, T)
        attn_weights = F.softmax(attn_scores, dim=1)  # (batch, T)
        
        # --- Weighted predictions ---
        y_pred = self.output_layer(features).squeeze(-1)  # (batch, T)
        y_pred = y_pred * attn_weights  # Weight by attention

        return y_pred, attn_weights

    def loss(self, model_output, data):
        # loss that ignores NaN in y

        loss = (model_output[0] - data["y"]).pow(2) * data["X_mask"].unsqueeze(-1)
        return loss.sum() / data["X_mask"].sum()


# Create dataset and loader
loader_train = utils.make_longitudinal_data_loader(data_train[0], data_train[1], measurement_ids=measurement_ids, n_timepoints=n_timepoints)
loader_val = utils.make_longitudinal_data_loader(data_val[0], data_val[1], measurement_ids=measurement_ids, n_timepoints=n_timepoints)
loader_test = utils.make_longitudinal_data_loader(data_test[0], data_test[1], measurement_ids=measurement_ids, n_timepoints=n_timepoints)

for batch in loader_train:
    print(batch['y'].shape)
    print(batch['X'].shape)
    print(batch['measurement_ids'].shape)
    print(batch['X_mask'].shape)

    break

# try
model = MaskedTimeAttentionModel(input_dim=p, time_steps=n_timepoints, n_measurements=n_measurements)
model(batch)


# 4. Training Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MaskedTimeAttentionModel(input_dim=p, time_steps=n_timepoints, n_measurements=n_measurements)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
model(batch)

# Plot model
from torchviz import make_dot

dummy_input = torch.randn(1, p)  # Batch size=1
# Forward pass to capture computation graph
y_pred, attn_weights = model(loader_test.collate_fn(loader_test.dataset))
make_dot(y_pred, params=dict(model.named_parameters())).render("model_architecture", format="png")

# 5. Training Loop
num_epochs = 300

trainer = utils.Training(loader_train, loader_val)
trainer.losses.keys()

trainer.training_loop(model, optimizer, num_epochs)

plt.plot(trainer.losses["train"], label="train")
plt.plot(trainer.losses["val"], label="val")
plt.legend()
plt.show()

# compare attention weights
with torch.no_grad():
    learned_weights = model(loader_test.collate_fn(loader_test.dataset))[1].numpy()  # Shape: (1, 5)
    true_weights = attn_weights[:1]    # Shape: (1, 5)
learned_weights.shape
attn_weights.shape

# standardize the weights and cluster using DBSCAN
scaler = StandardScaler()
attn_weights_scaled = scaler.fit_transform(learned_weights)

import hdbscan
# Cluster with HDBSCAN
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=10,           # Smallest group size
    min_samples=5,                # Controls density sensitivity
    metric='euclidean',           # Distance metric
    cluster_selection_method='eom' # Excess of mass (vs 'leaf')
)
clusters = clusterer.fit_predict(attn_weights_scaled)

# Number of clusters (excluding noise, labeled as -1)
n_clusters = len(np.unique(clusters))
print(f"Found {n_clusters} clusters")

# Adding UMAP for dimension reduction
import umap.umap_ as umap

reducer = umap.UMAP(n_components=2, random_state=42)
embedding = reducer.fit_transform(attn_weights_scaled)

# Plot clusters
plt.scatter(
    embedding[:, 0], embedding[:, 1],
    c=clusters, cmap='Spectral', alpha=1.
)
plt.colorbar(label='Cluster')
plt.title("UMAP Projection of Attention Weights")
plt.show()

df = pd.DataFrame({
    'patient_id': np.arange(n_test),
    'cluster': clusters,
    **{f'y_t{t}': data_test[1][:, t] for t in range(n_timepoints)}
})

# Compute mean y trajectory per cluster
mean_trajectories = df.iloc[:, 1:].groupby('cluster').mean()  # Shape: [n_clusters, T_timepoints]

plt.figure(figsize=(10, 6))
for cluster in mean_trajectories.index:
    if cluster == -1:
        label = f'{cluster} - outliers'
    else:
        label = cluster
    plt.plot(
        range(T_timepoints),
        mean_trajectories.loc[cluster],
        label=f'Cluster {label}',
        marker='o'
    )
plt.xlabel('Time')
plt.ylabel('Mean Outcome (y)')
plt.title('Temporal Patterns of y by Attention Cluster')
plt.legend()
plt.show()
