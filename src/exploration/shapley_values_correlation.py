# Shapley values with and without standardization
import os
import copy

import shap
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.utils import training_wrapper
from src.utils import data_loading_wrappers


# generate linear regression data
n = 100
p = 4

# if correlated
np.random.seed(34)
X = np.random.randn(n, p)

cov_matrix = np.zeros([p, p])
np.fill_diagonal(cov_matrix, 1.)

# cor x1, x2
cov_matrix[0, 1] = cov_matrix[1, 0] = 0.9

X = X.dot(np.linalg.cholesky(cov_matrix).transpose())
y = X.dot(np.ones(p)) + np.random.randn(n) * 0.2
y = y[..., None]

# ----------------------- OLS ------------------------
np.linalg.inv(X.transpose().dot(X)).dot(X.transpose().dot(y))

# ----------------------- NN ------------------------
class LinearModel(nn.Module):
    def __init__(self, input_dim: int):
        super(LinearModel, self).__init__()
        # self.fc1 = nn.Linear(input_dim, 5)
        # self.fc2 = nn.Linear(5, 1)
        self.fc1 = nn.Linear(input_dim, 1)

    def forward(self, batch):
        # return self.fc2(self.fc1(batch[0]))
        return self.fc1(batch[0])

    def loss(self, m_out, batch):
        # label prediction loss
        return [nn.functional.mse_loss(m_out, batch[1], reduction='mean')]


# get tensors
X_tensor = torch.FloatTensor(X).to(torch.device("cpu"))

y_tensor = torch.FloatTensor(y).to(torch.device("cpu"))
data_tensor = [X_tensor, y_tensor]

# make tensor data loaders
dataloader = data_loading_wrappers.make_data_loader(*data_tensor, batch_size=50)

# 4. Training Setup
device = torch.device("cpu")
model_nn = LinearModel(p).to(device)
optimizer = optim.Adam(model_nn.parameters(), lr=1e-3)

# Training Loop
num_epochs = 1500

trainer = training_wrapper.Training(dataloader)
trainer.training_loop(model_nn, optimizer, num_epochs)

model_nn.fc1.weight

# SHAP explanation
class EnsembleModel(torch.nn.Module):
    def __init__(self, model):
        super(EnsembleModel, self).__init__()
        self.model = model

    def forward(self, *x):
        x_list = list(x)
        self.model.eval()
        output = self.model(x_list)
        return output


# shap model
model_shap = EnsembleModel(model_nn)

base_value = model_shap(*data_tensor).mean().detach().item()
explainer = shap.GradientExplainer(model_shap, X_tensor)

shap_values = explainer.shap_values(X_tensor)
shap_values.shape
shap_values = shap_values[..., -1]

shap.summary_plot(
    shap_values,
    features=X_tensor,
    show=True
)
shap_values_std = shap_values.std(axis=0)
shap_values_std

row_id = 0
explanation = shap.Explanation(
    values=shap_values[row_id, :],
    base_values=base_value,
    data=X[row_id, :]
)
shap.plots.waterfall(explanation, show=True)

# new samples - 3 different observations
new_data = torch.tensor(np.array([
    [1., 1.1, -1., 0.5],
    ]),
    dtype=torch.float32
)

model_nn([new_data, new_data])
shap_values = explainer.shap_values(new_data)
shap_values.shape
shap_values = shap_values[..., -1]

np.round(shap_values, 3)
# array([[ 0.908,  0.97 , -1.024,  0.471]])

row_id = 2
explanation = shap.Explanation(
    values=shap_values[row_id, :],
    base_values=base_value,
    data=new_data[row_id, :].numpy()
)
shap.plots.waterfall(explanation, show=True)
# one standard deviation corresponds to a change according to the estimated shapley value


# --------------------------------------------------
# Try with conditional expectations
# Sample background data from the conditional distribution 
Sigma_1 = cov_matrix[0:1, 0:1]
Sigma_2 = cov_matrix[1:, 1:]
Sigma_12 = cov_matrix[0:1, 1:]
Sigma_21 = Sigma_12.transpose()

# fix a value for x1
x1 = new_data[0][0].item()

mu_cond = Sigma_12 * 1/Sigma_1 * (x1)
cov_cond = Sigma_2 - Sigma_21 * 1/Sigma_1 * Sigma_12

np.random.seed(34)
X_cond = np.random.randn(n, p - 1)
X_cond = X_cond.dot(np.linalg.cholesky(cov_cond).transpose()) + mu_cond
X_cond = np.concatenate([np.ones([n, 1]) * x1, X_cond], axis=1)
X_cond_tensor = torch.tensor(X_cond, dtype=torch.float32)

# calculate shap values
cond_base_value = model_shap(*[X_cond_tensor, X_cond_tensor]).mean().detach().item()
explainer = shap.GradientExplainer(model_shap, X_cond_tensor)

cond_shap_values = explainer.shap_values(new_data)
cond_shap_values.shape
cond_shap_values = cond_shap_values[..., -1]
np.round(cond_shap_values, 3)


# ----------------------------------------------------------------------------
# ----------------------- Manual Gradient Explainer --------------------------
# ----------------------------------------------------------------------------
def gradient_shap_minimal(model, background_data, x_explain, n_steps=10):
    """
    Minimal implementation of GradientExplainer (Expected Gradients).
    
    Args:
        model: PyTorch nn.Module (must be differentiable)
        background_data: (n_background, n_features) reference dataset as torch.Tensor
        x_explain: (n_features,) instance to explain as torch.Tensor
        n_steps: number of steps in path integral
        
    Returns:
        shap_values: (n_features,) SHAP values
        baseline: expected value
    """
    
    # Add batch dimension if needed
    if x_explain.dim() == 1:
        x_explain = x_explain.unsqueeze(0)
    
    # 1. Baseline: expected value over background data
    with torch.no_grad():
        baseline = model(background_data).mean().item()
    
    # 2. Sample baselines from background data
    n_background = len(background_data)
    indices = torch.randperm(n_background)
    
    baselines = background_data[indices]
    x_explain_rep = x_explain.repeat(n_background, 1)  # Repeat for each baseline
    
    # 3. Create interpolation paths
    alphas = torch.linspace(0, 1, n_steps + 1)[1:]  # Skip alpha=0
    alphas = alphas.view(-1, 1, 1)  # Shape: (n_steps, 1, 1) for broadcasting
    
    # Initialize attribution accumulator
    attribution = torch.zeros_like(x_explain)
    
    # 4. Compute path integral for each baseline
    for i, baseline_i in enumerate(baselines):
        # Interpolation: x(α) = baseline + α * (x - baseline)
        # We'll compute gradients at each α
        
        for alpha in alphas:
            # Interpolated point
            x_interp = baseline_i + alpha * (x_explain_rep[i] - baseline_i)
            x_interp.requires_grad_(True)
            
            # Forward pass
            output = model(x_interp.unsqueeze(0))
            
            # Backward pass
            if output.dim() == 0:
                output.backward()
            else:
                output.sum().backward()
            
            # Expected Gradients formula: (x - baseline) * gradient
            # Use trapezoidal rule weights
            if alpha == alphas[0]:
                weight = 0.5 * alpha.item()  # First point
            elif alpha == alphas[-1]:
                weight = 0.5 * (alpha.item() - alphas[-2].item())  # Last point
            else:
                weight = alpha.item() - alphas[torch.where(alphas == alpha)[0][0] - 1].item()  # Middle
            
            attribution += weight * (x_explain_rep[i] - baseline_i) * x_interp.grad
            
            # Clean up
            x_interp.grad = None
    
    # 5. Average over baselines and steps
    shap_value = attribution / n_background
    
    return shap_value.detach().numpy(), baseline


gradient_shap_minimal(
    model=model_nn,
    background_data=X_tensor,
    x_explain=new_data,
    n_steps=100
)

explainer = shap.GradientExplainer(model_shap, X_tensor)
cond_shap_values = explainer.shap_values(new_data)
cond_shap_values.shape
cond_shap_values = cond_shap_values[..., -1]
np.round(cond_shap_values, 3)

# check single steps
new_data = [0.9, 1.1, -0.9, 0.5]

new_data = torch.tensor(np.array([
    new_data,
    ]),
    dtype=torch.float32
)

baseline_i = torch.tensor(np.array([
    [1.0, -1.0, -0.5, 0.],
    ]),
    dtype=torch.float32
)

n_steps = 100
alphas = torch.linspace(0, 1, n_steps + 1)[1:]  # Skip alpha=0
alphas = alphas.view(-1, 1, 1)  # Shape: (n_steps, 1, 1) for broadcasting

attribution = torch.zeros_like(new_data)

# 4. Compute path integral for each baseline
# Interpolation: x(α) = baseline + α * (x - baseline)
# We'll compute gradients at each α

for alpha in alphas:
    # Interpolated point
    x_interp = baseline_i + alpha * (new_data - baseline_i)
    x_interp.requires_grad_(True)
    
    # Forward pass
    output = model_nn(x_interp.unsqueeze(0))
    
    # Backward pass
    if output.dim() == 0:
        output.backward()
    else:
        output.sum().backward()
    
    # Expected Gradients formula: (x - baseline) * gradient
    # Use trapezoidal rule weights
    if alpha == alphas[0]:
        weight = 0.5 * alpha.item()  # First point
    elif alpha == alphas[-1]:
        weight = 0.5 * (alpha.item() - alphas[-2].item())  # Last point
    else:
        weight = alpha.item() - alphas[torch.where(alphas == alpha)[0][0] - 1].item()  # Middle
    
    attribution += weight * (new_data - baseline_i) * x_interp.grad
    
    # Clean up
    x_interp.grad = None

attribution
