# Model with missing key predictors
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

import numpy as np
from scipy.stats import pearsonr

import matplotlib.pyplot as plt
import shap

import os
os.chdir("./src")

from vae_attention import training_wrapper

from utils import data_loading_wrappers
from utils.model_output_details import count_parameters
from utils import plots


torch.get_num_threads()
torch.set_num_threads(6)


# generate simple linear data
n_train = 100
n_test = 200
n_val = 200
n = n_train + n_val + n_test
p = 10
p1 = 5
p0 = 5
batch_size = 50

# custom W
np.random.seed(323425)
W = np.random.choice(
    [-1, -0.8, -0.5, 1, 0.8, 0.5],
    size=(p1)
)
beta = np.random.choice(
    [-1, 1],
    size=(p0)
)

X = np.random.randn(n, p)
y = np.dot(X[:, 0:p1], W) + np.dot(X[:, p1:p], beta) + np.random.randn(n) * 0.5

# get tensors
X_tensor = torch.FloatTensor(X).to(torch.device("cpu"))
y_tensor = torch.FloatTensor(y).to(torch.device("cpu"))

# split data
data_split = data_loading_wrappers.DataSplit(X.shape[0], test_size=n_test, val_size=n_val)
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
train_dataloader = data_loading_wrappers.make_data_loader(*tensor_data_train, batch_size=batch_size)
test_dataloader = data_loading_wrappers.make_data_loader(*tensor_data_test, batch_size=batch_size)
val_dataloader = data_loading_wrappers.make_data_loader(*tensor_data_val, batch_size=batch_size)

next(iter(train_dataloader))[0].shape  # X
next(iter(train_dataloader))[1].shape  # y


class LinearModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 1):
        super(LinearModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)

    def forward(self, batch):
        return self.fc1(batch[0])

    def loss(self, m_out, batch):
        # label prediction loss
        return nn.functional.mse_loss(m_out, batch[1], reduction='mean')


class NNModel(nn.Module):
    def __init__(self, input_dim: int, ldim: int, output_dim: int = 1):
        super(NNModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, ldim)
        self.fc2 = nn.Linear(ldim, output_dim)

    def forward(self, batch):
        x = torch.relu(self.fc1(batch[0]))
        pred = self.fc2(x)

        return pred

    def loss(self, m_out, batch):
        # label prediction loss
        return nn.functional.mse_loss(m_out, batch[1], reduction='mean')


# 4. Training Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = NNModel(p, ldim=50).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Coefficients
count_parameters(model)

# Training Loop
num_epochs = 200

trainer = training_wrapper.Training(train_dataloader, val_dataloader)
trainer.training_loop(model, optimizer, num_epochs)

plt.plot(trainer.losses["train"], label="train")
plt.plot(trainer.losses["val"], label="val")
plt.vlines(np.argmin(trainer.losses["val"]), 0, max(trainer.losses["val"]), color="red")
plt.vlines(np.argmin(trainer.losses["train"]), 0, max(trainer.losses["train"]), color="blue")
plt.hlines(np.min(trainer.losses["val"]), 0, len(trainer.losses["val"]), color="red", linestyles="--")
plt.hlines(np.min(trainer.losses["train"]), 0, len(trainer.losses["val"]), color="blue", linestyles="--")
plt.legend()
plt.show()


linear_model = LinearModel(p).to(device)
optimizer = optim.Adam(linear_model.parameters(), lr=1e-3)
# Coefficients
count_parameters(linear_model)

# Training Loop
num_epochs = 50

trainer = training_wrapper.Training(train_dataloader, val_dataloader)
trainer.training_loop(linear_model, optimizer, num_epochs)

plt.plot(trainer.losses["train"], label="train")
plt.plot(trainer.losses["val"], label="val")
plt.vlines(np.argmin(trainer.losses["val"]), 0, max(trainer.losses["val"]), color="red")
plt.vlines(np.argmin(trainer.losses["train"]), 0, max(trainer.losses["train"]), color="blue")
plt.hlines(np.min(trainer.losses["val"]), 0, len(trainer.losses["val"]), color="red", linestyles="--")
plt.hlines(np.min(trainer.losses["train"]), 0, len(trainer.losses["val"]), color="blue", linestyles="--")
plt.legend()
plt.show()


# -------------------------------------------------------
# ----------- Now removing 2 key predictors -----------
# -------------------------------------------------------

X1 = X[:, 0:p1]
X1.shape

X_tensor = torch.FloatTensor(X1).to(torch.device("cpu"))

# split data
data_split = data_loading_wrappers.DataSplit(X1.shape[0], test_size=n_test, val_size=n_val)
print("train: ", len(data_split.train_index), 
    "val: ", len(data_split.val_index),
    "test: ", len(data_split.test_index)
)

data_train = data_split.get_train(X1, y)
data_test = data_split.get_test(X1, y)
data_val = data_split.get_val(X1, y)

tensor_data_train = data_split.get_train(X_tensor, y_tensor)
tensor_data_test = data_split.get_test(X_tensor, y_tensor)
tensor_data_val = data_split.get_val(X_tensor, y_tensor)

# make tensor data loaders
train_dataloader = data_loading_wrappers.make_data_loader(*tensor_data_train, batch_size=batch_size)
test_dataloader = data_loading_wrappers.make_data_loader(*tensor_data_test, batch_size=batch_size)
val_dataloader = data_loading_wrappers.make_data_loader(*tensor_data_val, batch_size=batch_size)


# 4. Training Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = NNModel(p1, ldim=50).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
# Coefficients
count_parameters(model)

# Training Loop
num_epochs = 100

trainer = training_wrapper.Training(train_dataloader, val_dataloader)
trainer.training_loop(model, optimizer, num_epochs)

plt.plot(trainer.losses["train"], label="train")
plt.plot(trainer.losses["val"], label="val")
plt.vlines(np.argmin(trainer.losses["val"]), 0, max(trainer.losses["val"]), color="red")
plt.vlines(np.argmin(trainer.losses["train"]), 0, max(trainer.losses["train"]), color="blue")
plt.hlines(np.min(trainer.losses["val"]), 0, len(trainer.losses["val"]), color="red", linestyles="--")
plt.hlines(np.min(trainer.losses["train"]), 0, len(trainer.losses["val"]), color="blue", linestyles="--")
plt.legend()
plt.show()


linear_model = LinearModel(p1).to(device)
optimizer = optim.Adam(linear_model.parameters(), lr=1e-3)
# Coefficients
count_parameters(linear_model)

# Training Loop
num_epochs = 50

trainer = training_wrapper.Training(train_dataloader, val_dataloader)
trainer.training_loop(linear_model, optimizer, num_epochs)

plt.plot(trainer.losses["train"], label="train")
plt.plot(trainer.losses["val"], label="val")
plt.vlines(np.argmin(trainer.losses["val"]), 0, max(trainer.losses["val"]), color="red")
plt.vlines(np.argmin(trainer.losses["train"]), 0, max(trainer.losses["train"]), color="blue")
plt.hlines(np.min(trainer.losses["val"]), 0, len(trainer.losses["val"]), color="red", linestyles="--")
plt.hlines(np.min(trainer.losses["train"]), 0, len(trainer.losses["val"]), color="blue", linestyles="--")
plt.legend()
plt.show()

linear_model.fc1.weight
