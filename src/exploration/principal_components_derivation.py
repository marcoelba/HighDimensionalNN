# Principal components derivation
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.parametrizations import orthogonal
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import shap
from scipy.linalg import toeplitz
from scipy.stats import multivariate_normal
from sklearn.datasets import load_iris

import os
os.chdir("./src")

from model_utils import models, utils


# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

X = X[y < 2]
y = y[y < 2]
# center data
X = X - np.mean(X, axis=0)

n = X.shape[0]
p = X.shape[1]

n_test = 20
n_val = n_test

# split train test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=n_test)
X_train.shape

X_val = X_test.copy()
y_val = y_test.copy()

# Convert numpy arrays to PyTorch tensors
X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).float()[..., None]  # or .long() for class labels
X_test_tensor = torch.from_numpy(X_test).float()
y_test_tensor = torch.from_numpy(y_test).float()[..., None]
X_val_tensor = torch.from_numpy(X_val).float()
y_val_tensor = torch.from_numpy(y_val).float()[..., None]  # or .long() for class labels


# crete data loaders
batch_size = 32
train_loader = utils.make_data_loader(X_train_tensor, y_train_tensor)
val_loader = utils.make_data_loader(X_val_tensor, y_val_tensor)
test_loader = utils.make_data_loader(X_test_tensor, y_test_tensor)


# Inspect a batch
for batch_X, batch_y in train_loader:
    print(f"Batch X shape: {batch_X.shape}")  # (batch_size, num_features)
    print(f"Batch y shape: {batch_y.shape}")  # (batch_size,) or (batch_size, num_targets)
    break


# --------------- PCA ------------------------
n_components = 2

pca = PCA(n_components=n_components)
pca.fit(X_train)
print(pca.explained_variance_ratio_)

pca_X_test_pred = pca.transform(X_test)

plt.scatter(pca_X_test_pred[:, 0], pca_X_test_pred[:, 1], c=y_test, cmap='viridis', edgecolor='k')
plt.title('PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

pca.components_


# -------------------- My Model -----------------
# Define loss and optimiser and model
model = MixModel(p, n_components)

# test
model(X_test_tensor[1:5, :])[0].shape
model(X_test_tensor[1:5, :])[1].shape

# train loop
criterion = nn.MSELoss()
# , weight_decay=1e-5
optimizer = optim.Adam(model.parameters(), lr=0.01)

n_epochs = 100

train_losses = []
val_losses = []

for epoch in range(n_epochs):
    epoch_train_loss = 0.0

    model.train(True)
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        dec, enc = model(data)
        
        loss = criterion(dec, data)

        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}')
        
        epoch_train_loss += loss.item()

    # Average epoch loss and store it
    train_losses.append(epoch_train_loss / len(train_loader))

    # Validation phase
    model.eval()
    epoch_val_loss = 0.0

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            dec, enc = model(batch_X)

            loss = criterion(dec, batch_X)
            epoch_val_loss += loss.item()

    # Average epoch loss and store it
    val_losses.append(epoch_val_loss / len(val_loader))

# check the loss
plt.plot(train_losses, label='Train Loss', ls="-")

plt.plot(val_losses, label='Val Loss', ls="-")

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curves')
plt.legend()
plt.show()

# params
for name, param in model.named_parameters():
    print(name, param)

coeffs = model.enc1.weight.detach().numpy()
coeffs = model.dec1.weight.detach().numpy()[0]

plt.plot(coeffs, color='green', marker='o', linestyle='')
plt.show()


# model predictions
# 1. Set model to evaluation mode
model.eval()

# 2. Generate predictions (no gradients)
with torch.no_grad():
    test_dec, test_enc = model(X_test_tensor)

# 3. Calculate RMSE
mse = nn.MSELoss()
rmse = torch.sqrt(mse(test_dec, X_test_tensor)).item()
print(f"RMSE: {rmse:.4f}")


plt.scatter(test_enc.numpy()[:, 0], test_enc.numpy()[:, 1], c=y_test, cmap='viridis', edgecolor='k')
plt.scatter(pca_X_test_pred[:, 0], pca_X_test_pred[:, 1], c=y_test, cmap='viridis', edgecolor='k', marker="X")

plt.title('PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

np.corrcoef(pca_X_test_pred, rowvar=False)
np.corrcoef(test_enc.numpy(), rowvar=False)


# Clusters in lower dimensional space
kmeans_pca = KMeans(n_clusters=2).fit(pca_X_test_pred)
print(kmeans_pca.labels_)

kmeans_ae = KMeans(n_clusters=2).fit(test_enc.numpy())
print(kmeans_ae.labels_)


# Now perform logistic regression

# PCA
from sklearn.linear_model import LogisticRegression
pca_X = pca.transform(X_train)
pca_X_test = pca.transform(X_test)

clf = LogisticRegression(random_state=0, penalty=None).fit(pca_X, y_train)
pca_pred = clf.predict(pca_X_test)
clf.score(pca_X, y_train)


# Autoencoder
with torch.no_grad():
    ae_X = model(X_train_tensor)[1]
    ae_X_test = model(X_test_tensor)[1]
    ae_X_val = model(X_val_tensor)[1]


class LogisticModel(nn.Module):
    def __init__(self, input_dim: int):
        super(LogisticModel, self).__init__()
        # encoder layers
        self.l1 = nn.Linear(input_dim, 1)

    def forward(self, x):
        # encoder-decoder
        x = torch.sigmoid(self.l1(x))

        return x


# Initialize datasets
train_loader = make_data_loader(ae_X, y_train_tensor, batch_size = 32)
val_loader = make_data_loader(ae_X_val, y_val_tensor, batch_size = 32)
test_loader = make_data_loader(ae_X_test, y_test_tensor, batch_size = 32)


model_logistic = LogisticModel(2)

# test
model_logistic(ae_X[1:5, :])

# train loop
criterion = nn.BCEWithLogitsLoss()
# , weight_decay=1e-5
optimizer = optim.Adam(model_logistic.parameters(), lr=0.01)

n_epochs = 200
train_losses = []
val_losses = []

for epoch in range(n_epochs):
    epoch_train_loss = 0.0

    model_logistic.train(True)
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        out = model_logistic(data)
        
        loss = criterion(out, target)

        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}')
        
        epoch_train_loss += loss.item()

    # Average epoch loss and store it
    train_losses.append(epoch_train_loss / len(train_loader))

    # Validation phase
    model_logistic.eval()
    epoch_val_loss = 0.0

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            out = model_logistic(batch_X)

            loss = criterion(out, batch_y)
            epoch_val_loss += loss.item()

    # Average epoch loss and store it
    val_losses.append(epoch_val_loss / len(val_loader))

# check the loss
plt.plot(train_losses, label='Train Loss', ls="-")

plt.plot(val_losses, label='Val Loss', ls="-")

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curves')
plt.legend()
plt.show()

# params
for name, param in model_logistic.named_parameters():
    print(name, param)


# model predictions
# 1. Set model to evaluation mode
model_logistic.eval()

# 2. Generate predictions (no gradients)
with torch.no_grad():
    pred = model_logistic(ae_X_test)

pred > 0.5
