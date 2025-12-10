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
p = 3

# if correlated
X = np.random.randn(n, p)
cov_matrix = np.zeros([p, p])
np.fill_diagonal(cov_matrix, 1.)
# cor x1, x2
cov_matrix[0, 1] = cov_matrix[1, 0] = 0.98

np.round(np.cov(X.dot(np.linalg.cholesky(cov_matrix).transpose()), rowvar=False), 3)
np.round(np.corrcoef(X.dot(np.linalg.cholesky(cov_matrix).transpose()), rowvar=False), 3)

X = X.dot(np.linalg.cholesky(cov_matrix).transpose())
y = X.dot([1., 0., 1.]) + np.random.randn(n) * 0.2
y = y[..., None]

# ----------------------- OLS ------------------------
np.linalg.inv(X.transpose().dot(X)).dot(X.transpose().dot(y))
# ~[1., .1, .1]

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
num_epochs = 2000

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
    [0., 0., 0.],
    [1., 1.1, -1.],
    [2., 1.9, 0.]
    ]),
    dtype=torch.float32
)

model_nn([new_data, new_data])
shap_values = explainer.shap_values(new_data)
shap_values.shape
shap_values = shap_values[..., -1]

row_id = 2
explanation = shap.Explanation(
    values=shap_values[row_id, :],
    base_values=base_value,
    data=new_data[row_id, :].numpy()
)
shap.plots.waterfall(explanation, show=True)
# one standard deviation corresponds to a change according to the estimated shapley value


# --------------------------------------------------------------------------
# Using Kernel Shap, which does not account for correlation in the features
# --------------------------------------------------------------------------

# SHAP explanation
def model_shap(x):
    x = torch.Tensor(x)
    pred = model_nn([x, x]).detach().numpy()
    return pred

model_shap(X)

base_value = model_shap(X).mean()

explainer = shap.KernelExplainer(model_shap, X)
shap_values = explainer.shap_values(X)
shap_values.shape
shap_values = shap_values[..., -1]

shap.summary_plot(
    shap_values,
    features=X,
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


shap_values = explainer.shap_values(new_data.numpy())
shap_values.shape
shap_values = shap_values[..., -1]

row_id = 2
explanation = shap.Explanation(
    values=shap_values[row_id, :],
    base_values=base_value,
    data=new_data.numpy()[row_id, :]
)
shap.plots.waterfall(explanation, show=True)
