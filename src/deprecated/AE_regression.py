# AE + regression
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
from sklearn.preprocessing import StandardScaler

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

# add covariate covariates
X_rand = np.random.randn(n, 20)
X_rand.shape
X = np.concatenate([X, X_rand], axis=1)
X.shape
p = p + 20

n_test = 20
n_val = n_test

# ------------ random data ------------
n = 150
p0 = 20
p1 = 5
p = p0 + p1

X, y = utils.generate_data(n, p0, p1, corr_factor=0.8, sd_residuals=0.5)

n_test = 50
n_val = n_test

# split train test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=n_test)
X_train.shape

X_val = X_test.copy()
y_val = y_test.copy()

y_scaler = StandardScaler()
y_scaler.fit(y_train)

y_train = y_scaler.transform(y_train)
y_val = y_scaler.transform(y_val)
y_test = y_scaler.transform(y_test)


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


# Define loss and optimiser and model
n_components = 3

model = models.MixModel(p, n_components, activation=torch.sigmoid)

model = models.DeepMixModel(p, n_components, activation=None)

# test
model(X_test_tensor[1:5, :])
model(X_test_tensor[1:5, :])[1].shape

# train loop
criterion_pred = nn.BCEWithLogitsLoss()

criterion_pred = nn.MSELoss()

criterion_ae = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.005)
optimizer = optim.RMSprop(model.parameters(), lr=0.001)

n_epochs = 1000

train_losses = []
val_losses = []

ae_train_losses = []
ae_val_losses = []

for epoch in range(n_epochs):
    epoch_train_loss = 0.0
    epoch_ae_loss = 0.0

    model.train(True)
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        dec, enc, pred = model(data)
        
        loss_ae = criterion_ae(dec, data)
        loss_pred = criterion_pred(pred, target)
        loss = loss_ae + loss_pred

        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}')
        
        epoch_train_loss += loss.item()
        epoch_ae_loss += loss_ae.item()
    # Average epoch loss and store it
    train_losses.append(epoch_train_loss / len(train_loader))
    ae_train_losses.append(epoch_ae_loss / len(train_loader))

    # Validation phase
    model.eval()
    epoch_val_loss = 0.0
    ae_val_loss = 0.0

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            dec, enc, pred = model(batch_X)

            loss_ae = criterion_ae(dec, batch_X)
            loss_pred = criterion_pred(pred, batch_y)
            loss = loss_ae + loss_pred

            epoch_val_loss += loss.item()
            ae_val_loss += loss_ae.item()
    # Average epoch loss and store it
    val_losses.append(epoch_val_loss / len(val_loader))
    ae_val_losses.append(ae_val_loss / len(val_loader))

# check the loss
plt.plot(train_losses, label='Train Loss', ls="-")
plt.plot(ae_train_losses, label='AE Train Loss', ls="--")

plt.plot(val_losses, label='Val Loss', ls="-")
plt.plot(ae_val_losses, label='AE Val Loss', ls="--")

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curves')
plt.legend()
plt.show()

# params
for name, param in model.named_parameters():
    print(name, param)


# model predictions
# 1. Set model to evaluation mode
model.eval()

# 2. Generate predictions (no gradients)
with torch.no_grad():
    pred = model(X_test_tensor)

np.round(np.concat([pred[2].numpy(), y_test[..., None]], axis=1), 3)

np.round(np.concat([pred[2].numpy(), y_test], axis=1), 3)
np.corrcoef(pred[2].numpy(), y_test, rowvar=False)

plt.scatter(pred[2].numpy(), y_test)
plt.show()

# plot latent space
z = model(X_test_tensor)[1].detach().numpy()

plt.scatter(z[:, 0], z[:, 1], c=y_test_tensor.detach().numpy(), cmap='tab10')
plt.colorbar()
plt.show()


plt.scatter(z[:, 0], z[:, 2])
plt.show()


from sklearn.manifold import TSNE

# Reduce latent space to 2D
z_2d = TSNE(n_components=2).fit_transform(z)

# Plot colored by feature `i`
i = 1  # Feature index
plt.scatter(z_2d[:, 0], z_2d[:, 1], c=X_test[:, i], cmap='viridis')
plt.colorbar()
plt.title(f"Latent Space Colored by Feature {i}")
plt.show()


# plot weights latent space
encoder_weights = model.enc1.weight.detach().numpy()
encoder_weights.shape
plt.plot(encoder_weights[:, [1, 10]], marker='o')
plt.show()

# Saliency map

# Let `autoencoder` be your trained model and `x` a sample input
X_train_tensor.requires_grad_(True)  # Enable gradient tracking

# Get latent representation (z)
z = model(X_train_tensor)[1]

# For a specific latent dimension (e.g., z[0, 0])
z_dim = 1
latent_dim = z[0, z_dim]  # Select one dimension

# Compute gradient
latent_dim.backward()
saliency = X_train_tensor.grad.abs().mean(dim=0)  # Mean gradient magnitude per input feature

# Plot feature importance
print("Top influential features for latent dim", z_dim, ":", saliency.argsort(descending=True)[:5])


# -------------- SHAP explainer --------------
model.eval()
# backgrund samples
background = X_train

# Define a prediction function
def predict(x):
    x = torch.FloatTensor(x)
    with torch.no_grad():
        preds = model(x)
    return preds[2].numpy().flatten()

# Create KernelExplainer
explainer = shap.KernelExplainer(predict, background)  # Using 100 samples as background
# Deep explainer for NN
model.eval()
explainer = shap.DeepExplainer((model, model.fc1), X_train_tensor)

samples_to_explain = X_test
shap_values = explainer.shap_values(samples_to_explain)
shap_values.shape

# Plot feature importance
feature_names = [f'Feature {i}' for i in range(p)]

shap.summary_plot(shap_values, samples_to_explain, show = True)

shap.summary_plot(shap_values, samples_to_explain, plot_type="violin", show = True)
shap.summary_plot(shap_values, samples_to_explain, plot_type="compact_dot", show = True)


sample_ind = 1
shap.force_plot(explainer.expected_value, shap_values[sample_ind, :], samples_to_explain[sample_ind, :], 
    feature_names=[f'Feature {i}' for i in range(p1)], matplotlib=True
)

# For multiple predictions
shap.force_plot(explainer.expected_value, shap_values, samples_to_explain,
    feature_names=[f'Feature {i}' for i in range(p)], matplotlib=True
)

shap.decision_plot(explainer.expected_value, shap_values, X_test, feature_names=feature_names)

# dependence plot
shap.dependence_plot(0, shap_values, X_test, interaction_index=None)

shap.dependence_plot(0, shap_values, X_test, interaction_index='auto')

shap.dependence_plot(0, shap_values, X_test, interaction_index=2, cmap='coolwarm')

shap.dependence_plot(10, shap_values, X_test, interaction_index=11, cmap='coolwarm')

plt.scatter(shap_values[:, 0], shap_values[:, 2])
plt.show()


sample_ind = 1
shap.plots.waterfall(shap.Explanation(
    values=shap_values[sample_ind], 
    base_values=explainer.expected_value, 
    data=X_test[sample_ind], 
    feature_names=feature_names
    )
)

shap.plots.heatmap(shap.Explanation(
    values=shap_values, 
    base_values=explainer.expected_value, 
    data=X_test, 
    feature_names=feature_names
    )
)



shap.plots.violin(shap.Explanation(
    values=shap_values, 
    base_values=explainer.expected_value, 
    feature_names=feature_names)
)


# Using SAGE - Shapley Additive Global Explanations
import sage

def predict(x):
    x = torch.FloatTensor(x)
    with torch.no_grad():
        preds = model(x)
    return preds[2].numpy().flatten()


imputer = sage.MarginalImputer(predict, X_train)  # Background data
estimator = KernelEstimator(imputer, 'mse')
sage_values = estimator(X_test, y_test)
sage_values.plot(feature_names)
plt.show()

