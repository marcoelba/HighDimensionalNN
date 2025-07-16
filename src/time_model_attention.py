# latent space recovery
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.parametrizations import orthogonal
import torch.nn.functional as F

import numpy as np
from scipy.linalg import toeplitz
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
n_test = 100
n_val = 100
p = 10
T_timepoints = 5
noise_level = 0.1

# 3. Compute X = ZW + noise
X = np.random.normal(loc=0.0, scale=1.0, size=(n_patients, p))
X.shape

# Simulate Attention Weights, aka impact of features on time evolution
# Compute attention weights for each patient
attn_weights = np.zeros((n_patients, T_timepoints))

# Rule: Higher X[:,0] (severity) → Later time points matter more
for i in range(n_patients):
    if X[i, 0] > 0:  # High severity
        attn_weights[i] = [0.1, 0.2, 0.3, 0.2, 0.2]  # Late emphasis
    else:            # Low severity
        attn_weights[i] = [0.3, 0.3, 0.2, 0.1, 0.1]  # Early emphasis

# Add small random noise to weights
attn_weights += np.random.uniform(-0.05, 0.05, size=(n_patients, T_timepoints))
attn_weights = np.abs(attn_weights)  # Ensure positivity
attn_weights /= attn_weights.sum(axis=1, keepdims=True)  # Normalize to sum=1

# Baseline outcome (intercept - time 1) depends on X
baseline = 1 + 2 * X[:, 0] - 1 * X[:, 1] + 1 * X[:, 2]

# Time effect (linear trend)
time = np.arange(T_timepoints)
time_effect = 0.5 * time + 0.2 + np.sin(time)

# Patient-specific outcomes
y = np.zeros((n_patients, T_timepoints))

y_time = baseline[0] + time_effect + np.random.normal(0, 0.1, size=T_timepoints)
# Apply attention weights
y_time_att = y_time * attn_weights[0] * T_timepoints  # Scale to compensate for weight normalization
plt.plot(y_time, label="pre")
plt.plot(y_time_att, label="att")
plt.legend()
plt.show()


for i in range(n_patients):
    # Base trend + noise
    y[i] = baseline[i] + time_effect + np.random.normal(0, 0.1, size=T_timepoints)
    # Apply attention weights
    y[i] *= attn_weights[i] * T_timepoints  # Scale to compensate for weight normalization

# measurement noise
y += np.random.normal(0, noise_level, size=(n_patients, T_timepoints))

X.shape
y.shape


# Plot attention weights for two patients
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.bar(range(T_timepoints), attn_weights[0])  # High-severity patient
plt.title(f"Patient 0 (X[0]={X[0, 0]:.2f})")
plt.subplot(1, 2, 2)
plt.bar(range(T_timepoints), attn_weights[1])  # Low-severity patient
plt.title(f"Patient 1 (X[0]={X[1, 0]:.2f})")
plt.show()

plt.plot(y[0])
plt.plot(y[1])
plt.plot(y[5])
plt.show()

# get tensors
X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y)

# split data
data_split = utils.DataSplit(X.shape[0], test_size=n_test, val_size=n_val)
len(data_split.train_index)
len(data_split.test_index)
len(data_split.val_index)

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

for data in train_dataloader:
    print(data)
    break


# define model
class TimeAttentionModel(nn.Module):
    def __init__(self, input_dim, time_steps, hidden_dim=32):
        super().__init__()
        self.time_steps = time_steps

        self.time_embed = nn.Embedding(time_steps, 16)  # Time embeddings
        self.feature_mixer = nn.Linear(input_dim + 16, hidden_dim)  # X + time → features
        self.output_layer = nn.Linear(hidden_dim, 1)  # Predicts y(t) for one step
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # Processes X to compute attention scores
            nn.Tanh(),
            nn.Linear(hidden_dim, time_steps)  # Outputs attention weights per time step
        )
        
    def forward(self, x):
        batch_size = x.shape[0]
        time = torch.arange(self.time_steps).repeat(batch_size, 1).to(x.device)  # (batch, 5)
        time_embed = self.time_embed(time)  # (batch, 5, 16)
        
        # Expand X and mix with time
        # unsqueeze adds a dimension
        x_expanded = x.unsqueeze(1).repeat(1, self.time_steps, 1)  # (batch, 5, input_dim)
        x_time = torch.cat([x_expanded, time_embed], dim=-1)  # (batch, 5, input_dim + 16)
        
        features = self.feature_mixer(x_time)  # (batch, 5, hidden_dim)
        features = torch.relu(features)
        
        # Predict unweighted y(t) for all steps
        y_unweighted = self.output_layer(features).squeeze(-1)  # (batch, 5)
        
        # Compute attention weights (batch, time_steps)
        attn_scores = self.attention(x)  # (batch, 5)
        attn_weights = F.softmax(attn_scores, dim=1)  # Normalize to sum=1
        
        # Weight predictions by attention
        y_final = y_unweighted * attn_weights  # (batch, 5)
        
        return y_final, attn_weights  # Return both predictions and weights

    def loss(self, model_output, data):
        # Reconstruction loss (MSE)
        return nn.functional.mse_loss(model_output[0], data[1], reduction='mean')



# 4. Training Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TimeAttentionModel(p, time_steps=T_timepoints, hidden_dim=32)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
model(X_tensor)

# Plot model
# from torchviz import make_dot

# dummy_input = torch.randn(1, p)  # Batch size=1
# # Forward pass to capture computation graph
# y_pred, attn_weights = model(dummy_input)
# make_dot(y_pred, params=dict(model.named_parameters())).render("model_architecture", format="png")

# 5. Training Loop
num_epochs = 300

trainer = utils.Training(train_dataloader, val_dataloader, device=device)
trainer.losses.keys()

trainer.training_loop(model, optimizer, num_epochs)

plt.plot(trainer.losses["train"], label="train")
plt.plot(trainer.losses["val"], label="val")
plt.legend()
plt.show()

# compare attention weights
with torch.no_grad():
    learned_weights = model(tensor_data_test[0])[1].numpy()  # Shape: (1, 5)
    true_weights = attn_weights[:1]    # Shape: (1, 5)

print("Learned weights:", learned_weights[:1])
print("True weights:", true_weights)

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

# Cluster individuals and analyse the patterns of y
df = pd.DataFrame({
    'patient_id': np.arange(n_test),
    'cluster': clusters,
    **{f'y_t{t}': data_test[1][:, t] for t in range(T_timepoints)}
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


# Plot attention weights vs. y trajectories side-by-side
import seaborn as sns

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot 1: Mean attention weights per cluster
sns.heatmap(
    df.groupby('cluster').mean()[[f'y_t{t}' for t in range(T_timepoints)]],
    annot=True, cmap="YlGnBu", ax=ax1
)
ax1.set_title('Attention Weights by Cluster')

# Plot 2: Mean y trajectories per cluster
for cluster in mean_trajectories.index:
    if cluster == -1:
        label = f'{cluster} - outliers'
    else:
        label = cluster
    ax2.plot(
        range(T_timepoints),
        mean_trajectories.loc[cluster],
        label=f'Cluster {label}',
        marker='o'
    )
ax2.set_title('Outcome (y) Trajectories by Cluster')
ax2.legend()
plt.show()

# Use SHAP as well
