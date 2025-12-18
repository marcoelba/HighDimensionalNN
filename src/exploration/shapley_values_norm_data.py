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
p = 5

# covarites with different var
x1 = np.random.randn(n, 1)
x2 = np.random.randn(n, 1) * 10
x3 = np.random.randn(n, 1) * 0.1
x4 = np.random.randn(n, 1) * 0.1
x5 = np.random.randn(n, 1) * 0.1

X = np.concatenate([x1, x2, x3, x4, x5], axis=1)

y = X.dot(np.array([1., 1., 1., 1., 1.])) + np.random.randn(n) * 0.2
y = y[..., None]

# if correlated
X = np.random.randn(n, p)
cov_matrix = np.zeros([p, p])
np.fill_diagonal(cov_matrix, [1., 2., 0.1, 0.1, 0.1])
# cor x1, x2
cov_matrix[0, 1] = cov_matrix[1, 0] = 0.5
# cor x3, x4, x5
cov_matrix[2, 3] = cov_matrix[3, 2] = 0.05
cov_matrix[2, 4] = cov_matrix[4, 2] = 0.04
cov_matrix[3, 4] = cov_matrix[4, 3] = 0.04

np.round(np.cov(X.dot(np.linalg.cholesky(cov_matrix).transpose()), rowvar=False), 3)
np.round(np.corrcoef(X.dot(np.linalg.cholesky(cov_matrix).transpose()), rowvar=False), 3)

X = X.dot(np.linalg.cholesky(cov_matrix).transpose())
y = X.dot(np.array([1., 1., 1., 1., 1.])) + np.random.randn(n) * 0.2
y = y[..., None]

# ----------------------- OLS ------------------------
np.linalg.inv(X.transpose().dot(X)).dot(X.transpose().dot(y))
# ~[1., .1, .1]

# now with feature standardization
scaler = StandardScaler()

X_std = scaler.fit_transform(X)
np.linalg.inv(X_std.transpose().dot(X_std)).dot(X_std.transpose().dot(y))
# ~[1., 0.1, 10.]


# ----------------------- NN ------------------------
class LinearModel(nn.Module):
    def __init__(self, input_dim: int):
        super(LinearModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, batch):
        return self.fc2(self.fc1(batch[0]))

    def loss(self, m_out, batch):
        # label prediction loss
        return [nn.functional.mse_loss(m_out, batch[1], reduction='mean')]


# get tensors
X_tensor = torch.FloatTensor(X).to(torch.device("cpu"))
# or
X_tensor = torch.FloatTensor(X_std).to(torch.device("cpu"))

y_tensor = torch.FloatTensor(y).to(torch.device("cpu"))
data_tensor = [X_tensor, y_tensor]

# make tensor data loaders
dataloader = data_loading_wrappers.make_data_loader(*data_tensor, batch_size=50)

# 4. Training Setup
device = torch.device("cpu")
model_nn = LinearModel(p).to(device)
optimizer = optim.Adam(model_nn.parameters(), lr=1e-3)
optimizer = optim.SGD(model_nn.parameters(), lr=1e-4)

# Training Loop
num_epochs = 5000

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

# check single predictions
base_value
model_shap(*[data_tensor[0].mean(axis=0), data_tensor[0].mean(axis=0)])
model_nn([data_tensor[0].mean(axis=0), data_tensor[1].mean()])

row_id = 1
base_value + sum(shap_values[row_id, :])
data_tensor[1].mean() + sum(shap_values[row_id, :])

model_shap(*data_tensor)[row_id]

explanation = shap.Explanation(
    values=shap_values[row_id, :],
    base_values=base_value,
    data=X[row_id, :]
)
shap.plots.waterfall(explanation, show=True)

sum_shap_patient = np.abs(shap_values).sum(axis=1)
sum_shap_patient[np.argsort(sum_shap_patient)[::-1]]

# isolate features with small variance
small_std = np.std(shap_values, axis=0) < 0.5

shap.summary_plot(
    shap_values[:, small_std],
    features=X_tensor[:, small_std],
    show=True
)

row_id = 0
explanation = shap.Explanation(
    values=shap_values[row_id, small_std],
    base_values=base_value,
    data=X[row_id, small_std]
)
shap.plots.waterfall(explanation, show=True)


# new samples - 3 different observations
new_data = torch.tensor(np.array([[0., 0., 0.], [1., 0.1, -10.], [1., 1., -1.]]), dtype=torch.float32)

# with correlation
np.sqrt(cov_matrix)
new_data = torch.tensor(np.array([
    [0., 0., 0., 0., 0.],
    [1., 3., 0.3, 0.25, 0.35],
    [1., -3., 0.3, -0.3, 0.35]
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

norm_shap_values = shap_values / shap_values_std

row_id = 2
explanation = shap.Explanation(
    values=norm_shap_values[row_id, :],
    base_values=base_value,
    data=new_data[row_id, :].numpy()
)
shap.plots.waterfall(explanation, show=True)

np.abs(shap_values).mean(axis=0)
np.abs(shap_values)[:, 0].mean()
# split by sign
col = 2
shap_values[shap_values[:, col] > 0, col].mean()
shap_values[shap_values[:, col] < 0, col].mean()


####
T = 4
n = 5

x1 = [1, 1.5, 1.4, 1.1]
x2 = [1, 1.7, 1.6, 1.5]
x3 = [0.9, 1.1, 1.1, 0.9]
x4 = [0.9, 1.5, 1.0, 0.8]
x5 = [1.5, 2.0, 2.0, 1.7]
x6 = [1.5, 2.0, 2.2, 2.1]
x7 = [1., 1.6, 1.1, 1.2]

X = np.stack([
    x1, x2, x3, x4, x5, x6, x7
], axis=0)
X.shape

plt.plot(X.transpose())
plt.show()

baseline = X[:, 0:1]
X_net = (X - baseline)
sum_net = (X - baseline).sum(axis=1)

plt.plot(X_net.transpose(), label=np.round(np.abs(sum_net), 3))
plt.legend()
plt.show()

plt.plot(X.transpose(), label=np.round(np.abs(sum_net), 3))
plt.legend()
plt.show()
