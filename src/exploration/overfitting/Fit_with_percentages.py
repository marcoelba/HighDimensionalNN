# Checking fit with percentages
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

import numpy as np
from scipy.stats import pearsonr
from scipy.special import softplus
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import shap

from src.utils import training_wrapper
from src.utils import data_loading_wrappers
from src.utils.model_output_details import count_parameters
from src.utils import plots


class LinearModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 1):
        super(LinearModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)

    def forward(self, batch):
        return self.fc1(batch[0])

    def loss(self, m_out, batch):
        # label prediction loss
        return [nn.functional.mse_loss(m_out, batch[1], reduction='mean')]


class NNModel(nn.Module):
    def __init__(self, input_dim: int, ldim: int, dropout_prob, output_dim: int = 1):
        super(NNModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, ldim)
        self.fc2 = nn.Linear(ldim, output_dim)
        # self.dropout = nn.Dropout(p=dropout_prob)
        self.gelu = nn.GELU()

    def forward(self, batch):
        x = self.gelu(self.fc1(batch[0]))
        pred = self.fc2(x)

        return pred

    def loss(self, m_out, batch):
        # label prediction loss
        return [nn.functional.mse_loss(m_out, batch[1], reduction='mean')]


# generate data
n_train = 100
n_test = 200
n_val = 200
n = n_train + n_val + n_test
p1 = 8
p0 = 2
p = p1 + p0
batch_size = 50

# custom W
np.random.seed(323425)
W1 = np.random.choice(
    [-1, 1],
    size=(p1)
)
mean_covariates = np.random.choice([-1, -0.5, 0, 0.5, 1.], size=p1)
var_covariates = np.random.choice([0.1, 1., 2.], size=p1)
W0 = np.random.choice(
    [-1, 1],
    size=(p0)
)

X1 = np.random.randn(n, p1)
np.round(np.cov(X1, rowvar=False), 2)
cov_matrix = np.ones([p1, p1]) * 0.
np.fill_diagonal(cov_matrix, var_covariates)
X1 = X1.dot(np.linalg.cholesky(cov_matrix).transpose())
np.round(np.cov(X1, rowvar=False), 2)

# percentages
Xp = np.abs(np.random.randn(n, p0))
X = np.concatenate([X1, Xp], axis=1)

# y
y = np.dot(X1, W1) + np.dot(Xp, W0) + np.random.randn(n) * 0.6
y = y[..., None]

Xp = Xp / Xp.sum(axis=1)[..., None] * 100
X = np.concatenate([X, Xp[:, 0:1]], axis=1)

plt.hist(y)
plt.show()


# get tensors
X_tensor = torch.FloatTensor(X).to(torch.device("cpu"))
y_tensor = torch.FloatTensor(y).to(torch.device("cpu"))

# split data
data_split = data_loading_wrappers.DataSplit(X.shape[0], test_size=n_test, val_size=n_val)
print("train: ", len(data_split.train_index), 
    "val: ", len(data_split.val_index),
    "test: ", len(data_split.test_index)
)

tensor_data_train = data_split.get_train(X_tensor, y_tensor)
tensor_data_test = data_split.get_test(X_tensor, y_tensor)
tensor_data_val = data_split.get_val(X_tensor, y_tensor)


# ---------------------------------------------------------------------
# With preprocessing
# ---------------------------------------------------------------------

# features and outcome preprocessing
scaler_cov = StandardScaler()
scaler_cov.fit(X=tensor_data_train[0])
scaler_cov.mean_
scaler_cov.scale_

scaler_outcome = StandardScaler()
scaler_outcome.fit(X=tensor_data_train[1])
scaler_outcome.mean_
scaler_outcome.scale_

class TorchScaler:
    def __init__(self, scaler, dtype=torch.float32):
        self.mean = torch.tensor(scaler.mean_, dtype=dtype)
        self.scale = torch.tensor(scaler.scale_, dtype=dtype)

    def transform(self, x):
        """
            x: torch.Tensor
        """
        return (x - self.mean) / self.scale

    def inverse_transform(self, x):
        """
            x: torch.Tensor
        """
        return x * self.scale + self.mean


torch_scaler_outcome = TorchScaler(scaler_outcome)
torch_scaler_outcome.mean
torch_scaler_outcome.scale
torch_scaler_outcome.transform(tensor_data_train[1])
torch_scaler_outcome.inverse_transform(tensor_data_train[1])

torch_scaler_cov = TorchScaler(scaler_cov)
torch_scaler_cov.mean
torch_scaler_cov.scale
torch_scaler_cov.transform(tensor_data_train[0])
tensor_data_train[0][0:3] * torch_scaler_cov.scale + torch_scaler_cov.mean
scaler_cov.inverse_transform(tensor_data_train[0][0:3])

# apply scaling
preproc_tensor_data_train = [
    torch.tensor(scaler_cov.transform(tensor_data_train[0]), dtype=torch.float32),
    torch.tensor(scaler_outcome.transform(tensor_data_train[1]), dtype=torch.float32)
]
preproc_tensor_data_val = [
    torch.tensor(scaler_cov.transform(tensor_data_val[0]), dtype=torch.float32),
    torch.tensor(scaler_outcome.transform(tensor_data_val[1]), dtype=torch.float32)
]
preproc_tensor_data_test = [
    torch.tensor(scaler_cov.transform(tensor_data_test[0]), dtype=torch.float32),
    torch.tensor(scaler_outcome.transform(tensor_data_test[1]), dtype=torch.float32)
]

# make tensor data loaders
preproc_train_dataloader = data_loading_wrappers.make_data_loader(*preproc_tensor_data_train, batch_size=batch_size)
preproc_test_dataloader = data_loading_wrappers.make_data_loader(*preproc_tensor_data_test, batch_size=batch_size)
preproc_val_dataloader = data_loading_wrappers.make_data_loader(*preproc_tensor_data_val, batch_size=batch_size)


# 4. Training Setup
device = torch.device("cpu")
model_nn = NNModel(X.shape[1], ldim=10, dropout_prob=0.2).to(device)
optimizer = optim.Adam(model_nn.parameters(), lr=1e-3)

# Coefficients
count_parameters(model_nn)

# Training Loop
num_epochs = 300

trainer = training_wrapper.Training(preproc_train_dataloader, preproc_val_dataloader)
trainer.training_loop(model_nn, optimizer, num_epochs)

plt.plot(trainer.losses["train"], label="train")
plt.plot(trainer.losses["val"], label="val")
plt.vlines(np.argmin(trainer.losses["val"]), 0, max(trainer.losses["val"]), color="red")
plt.vlines(np.argmin(trainer.losses["train"]), 0, max(trainer.losses["train"]), color="blue")
plt.hlines(np.min(trainer.losses["val"]), 0, len(trainer.losses["val"]), color="red", linestyles="--")
plt.hlines(np.min(trainer.losses["train"]), 0, len(trainer.losses["val"]), color="blue", linestyles="--")
plt.legend()
plt.show()
# -----------------------

# SHAP explanations
class TensorModel(torch.nn.Module):
    def __init__(self, model):
        super(TensorModel, self).__init__()
        self.model = model
    
    def forward(self, x):
        """
        Args:
            x: torch tensor array with ALL features concatenated
        """
        x_list = [x]
        output = self.model(x_list)
        return output


model_shap = TensorModel(model_nn)
background_data = preproc_tensor_data_train[0]
model_shap(background_data)
explainer = shap.GradientExplainer(model_shap, background_data)
explain_data = preproc_tensor_data_test[0]

shap_values = explainer.shap_values(explain_data, return_variances=False)
shap_values.shape
shap_values = shap_values[..., -1]

shap.summary_plot(shap_values, explain_data, show = True)

# Get base values for the explainer
model_shap.eval()
with torch.no_grad():
    predictions_background = model_shap(background_data)
    base_value = predictions_background.numpy().mean()

feat_names = [f"f_{ii}" for ii in range(p+1)]

# model predictions on test data
model_shap.eval()
with torch.no_grad():
    predictions = model_shap(explain_data).numpy()

# Create Explanation object manually
ind = 1
explanation = shap.Explanation(
    values=shap_values[ind],  # For single output
    base_values=base_value,
    data=explain_data[ind].numpy(),  # Flatten if needed
    feature_names=feat_names
)

base_value + np.sum(shap_values[ind])

W
base_value
predictions[ind]
explain_data[ind]
shap.plots.waterfall(explanation, max_display=11)

sum(shap_values[ind] / scaler_cov.scale_)


# ---------------------------------------------------------------------
# WithOUT preprocessing on percentage
# ---------------------------------------------------------------------
scaler_cov = StandardScaler()
scaler_cov.fit(X=tensor_data_train[0])
scaler_cov.mean_[-1] = 0.0
scaler_cov.scale_[-1] = 1.0

scaler_outcome = StandardScaler()
scaler_outcome.fit(X=tensor_data_train[1])
scaler_outcome.mean_
scaler_outcome.scale_

# apply scaling
preproc_tensor_data_train = [
    torch.tensor(scaler_cov.transform(tensor_data_train[0]), dtype=torch.float32),
    torch.tensor(scaler_outcome.transform(tensor_data_train[1]), dtype=torch.float32)
]
preproc_tensor_data_val = [
    torch.tensor(scaler_cov.transform(tensor_data_val[0]), dtype=torch.float32),
    torch.tensor(scaler_outcome.transform(tensor_data_val[1]), dtype=torch.float32)
]
preproc_tensor_data_test = [
    torch.tensor(scaler_cov.transform(tensor_data_test[0]), dtype=torch.float32),
    torch.tensor(scaler_outcome.transform(tensor_data_test[1]), dtype=torch.float32)
]

# make tensor data loaders
preproc_train_dataloader = data_loading_wrappers.make_data_loader(*preproc_tensor_data_train, batch_size=batch_size)
preproc_test_dataloader = data_loading_wrappers.make_data_loader(*preproc_tensor_data_test, batch_size=batch_size)
preproc_val_dataloader = data_loading_wrappers.make_data_loader(*preproc_tensor_data_val, batch_size=batch_size)


# 4. Training Setup
device = torch.device("cpu")
model_nn_2 = NNModel(X.shape[1], ldim=10, dropout_prob=0.2).to(device)
optimizer = optim.Adam(model_nn_2.parameters(), lr=1e-3)

# Training Loop
num_epochs = 300

trainer = training_wrapper.Training(preproc_train_dataloader, preproc_val_dataloader)
trainer.training_loop(model_nn_2, optimizer, num_epochs)

plt.plot(trainer.losses["train"], label="train")
plt.plot(trainer.losses["val"], label="val")
plt.vlines(np.argmin(trainer.losses["val"]), 0, max(trainer.losses["val"]), color="red")
plt.vlines(np.argmin(trainer.losses["train"]), 0, max(trainer.losses["train"]), color="blue")
plt.hlines(np.min(trainer.losses["val"]), 0, len(trainer.losses["val"]), color="red", linestyles="--")
plt.hlines(np.min(trainer.losses["train"]), 0, len(trainer.losses["val"]), color="blue", linestyles="--")
plt.legend()
plt.show()
# -----------------------

# SHAP explanations
class TensorModel(torch.nn.Module):
    def __init__(self, model):
        super(TensorModel, self).__init__()
        self.model = model
    
    def forward(self, x):
        """
        Args:
            x: torch tensor array with ALL features concatenated
        """
        x_list = [x]
        output = self.model(x_list)
        return output


model_shap = TensorModel(model_nn_2)
background_data = preproc_tensor_data_train[0]
model_shap(background_data)
explainer = shap.GradientExplainer(model_shap, background_data)
explain_data = preproc_tensor_data_test[0]

shap_values = explainer.shap_values(explain_data, return_variances=False)
shap_values.shape
shap_values = shap_values[..., -1]

shap.summary_plot(shap_values, explain_data, show = True)

# Get base values for the explainer
model_shap.eval()
with torch.no_grad():
    predictions_background = model_shap(background_data)
    base_value = predictions_background.numpy().mean()

feat_names = [f"f_{ii}" for ii in range(p+1)]

# model predictions on test data
model_shap.eval()
with torch.no_grad():
    predictions = model_shap(explain_data).numpy()

# Create Explanation object manually
ind = 1
explanation = shap.Explanation(
    values=shap_values[ind],  # For single output
    base_values=base_value,
    data=explain_data[ind].numpy(),  # Flatten if needed
    feature_names=feat_names
)

base_value + np.sum(shap_values[ind])

W
base_value
predictions[ind]
explain_data[ind]
shap.plots.waterfall(explanation, max_display=11)

sum(shap_values[ind] / scaler_cov.scale_)

# -----------------------------------------------------------
