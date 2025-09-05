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

from utils import training_wrapper

from utils import data_loading_wrappers
from utils.model_output_details import count_parameters
from utils import plots


torch.get_num_threads()
torch.set_num_threads(6)


class LinearModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 1):
        super(LinearModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)

    def forward(self, batch):
        return self.fc1(batch[0])

    def loss(self, m_out, batch):
        # label prediction loss
        return nn.functional.mse_loss(m_out, batch[1], reduction='sum')


class NNModel(nn.Module):
    def __init__(self, input_dim: int, ldim: int, dropout_prob, output_dim: int = 1):
        super(NNModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, ldim)
        self.fc2 = nn.Linear(ldim, output_dim)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, batch):
        x = torch.relu(self.fc1(batch[0]))
        x = self.dropout(x)
        pred = self.fc2(x)

        return pred

    def loss(self, m_out, batch):
        # label prediction loss
        return nn.functional.mse_loss(m_out, batch[1], reduction='sum')


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
y = y[..., None]

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


# 4. Training Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = NNModel(p, ldim=50, dropout_prob=0.2).to(device)
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
optimizer = optim.RMSprop(linear_model.parameters(), lr=1e-2)
# Coefficients
count_parameters(linear_model)
linear_model(tensor_data_train)

# Training Loop
num_epochs = 250

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
np.linalg.inv(np.dot(X.transpose(), X)).dot(X.transpose()).dot(y)

# perfect prediction
y_pred = np.dot(X[:, 0:p1], W) + np.dot(X[:, p1:p], beta)
np.mean((y_pred - y.flatten())**2)


# -------------------------------------------------------
# ----------- Now removing 2 key predictors -------------
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

model1 = NNModel(p1, ldim=50, dropout_prob=0.2).to(device)
optimizer1 = optim.Adam(model1.parameters(), lr=1e-3)
# Coefficients
count_parameters(model1)

# Training Loop
num_epochs = 1000

trainer = training_wrapper.Training(train_dataloader, val_dataloader, noisy_gradient=False)
trainer.training_loop(model1, optimizer1, num_epochs, gradient_noise_std=0.0)

plt.plot(trainer.losses["train"], label="train")
plt.plot(trainer.losses["val"], label="val")
plt.vlines(np.argmin(trainer.losses["val"]), 0, max(trainer.losses["val"]), color="red")
plt.vlines(np.argmin(trainer.losses["train"]), 0, max(trainer.losses["train"]), color="blue")
plt.hlines(np.min(trainer.losses["val"]), 0, len(trainer.losses["val"]), color="red", linestyles="--")
plt.hlines(np.min(trainer.losses["train"]), 0, len(trainer.losses["val"]), color="blue", linestyles="--")
plt.legend()
plt.show()


linear_model1 = LinearModel(p1).to(device)
optimizer1 = optim.Adam(linear_model1.parameters(), lr=1e-2)
# Coefficients
count_parameters(linear_model1)

# Training Loop
num_epochs = 350

trainer = training_wrapper.Training(train_dataloader, val_dataloader)
trainer.training_loop(linear_model1, optimizer1, num_epochs)

plt.plot(trainer.losses["train"], label="train")
plt.plot(trainer.losses["val"], label="val")
plt.vlines(np.argmin(trainer.losses["val"]), 0, max(trainer.losses["val"]), color="red")
plt.vlines(np.argmin(trainer.losses["train"]), 0, max(trainer.losses["train"]), color="blue")
plt.hlines(np.min(trainer.losses["val"]), 0, len(trainer.losses["val"]), color="red", linestyles="--")
plt.hlines(np.min(trainer.losses["train"]), 0, len(trainer.losses["val"]), color="blue", linestyles="--")
plt.legend()
plt.show()

linear_model1.fc1.weight

np.mean((y - y.mean())**2)


# ----------------------------------------------------------------
# ------------ Using an ensemble model for prediction ------------
# ----------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X1 = X[:, 0:p1]
X1.shape

X_tensor = torch.FloatTensor(X1).to(torch.device("cpu"))
y_tensor = torch.FloatTensor(y).to(torch.device("cpu"))

# split data
data_split = data_loading_wrappers.DataSplit(X1.shape[0], test_size=n_test, val_size=n_val)
print("train: ", len(data_split.train_index), 
    "val: ", len(data_split.val_index),
    "test: ", len(data_split.test_index)
)

data_train = data_split.get_train(X1, y)
data_test = data_split.get_test(X1, y)

tensor_data_train = data_split.get_train(X_tensor, y_tensor)
tensor_data_test = data_split.get_test(X_tensor, y_tensor)

# make only test data loaders
test_dataloader = data_loading_wrappers.make_data_loader(*tensor_data_test, batch_size=batch_size)

# k-fold Cross-Validation
n_folds = 10

all_train_losses = []
all_val_losses = []
all_predictions = []
all_true = []
all_models = []

train_indices = np.random.permutation(np.arange(0, n_train))
# Split into k folds
folds = np.array_split(train_indices, n_folds)
num_epochs = 500

for fold in range(n_folds):
    print(f"Running k-fold validation on fold {fold+1} of {n_folds}")

    train_mask = torch.ones(n_train, dtype=torch.bool)
    train_mask[folds[fold]] = False

    # make train and validation data loader for the LOO cross-validation
    tensor_train_loo = [
        tensor_data_train[0][train_mask],
        tensor_data_train[1][train_mask]
    ]
    tensor_val_loo = [
        tensor_data_train[0][~train_mask],
        tensor_data_train[1][~train_mask],
    ]
    train_dataloader = data_loading_wrappers.make_data_loader(*tensor_train_loo, batch_size=batch_size)
    val_dataloader = data_loading_wrappers.make_data_loader(*tensor_val_loo, batch_size=1)

    # 4. Training Setup
    model = NNModel(p1, ldim=50, dropout_prob=0.2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training Loop
    trainer = training_wrapper.Training(train_dataloader, val_dataloader, noisy_gradient=False)
    trainer.training_loop(model, optimizer, num_epochs, gradient_noise_std=0.0)

    # use the model at the best validation iteration
    model.load_state_dict(trainer.best_model.state_dict())
    all_models.append(model)
    
    # Validate
    model.eval()
    with torch.no_grad():
        pred = model(tensor_val_loo)
        all_predictions.append(pred.numpy())
        all_true.append(tensor_val_loo[1].numpy())
    
    # store
    all_train_losses.append(np.min(trainer.losses["train"]))
    all_val_losses.append(np.min(trainer.losses["val"]))


predictions = np.concatenate(all_predictions, axis=0)
predictions.shape
ground_truth = np.concatenate(all_true, axis=0)
ground_truth.shape

torch.nn.functional.mse_loss(torch.tensor(predictions), torch.tensor(ground_truth), reduction='mean')

# Predictions on test data using all models
ensemble_predictions = np.zeros([n_test, n_folds])

for (fold, model) in enumerate(all_models):
    model.eval()
    with torch.no_grad():
        pred = model(tensor_data_test).numpy()
        ensemble_predictions[:, fold] = pred.squeeze()
# check predictions variability
ensemble_predictions[0]
ensemble_min = np.min(ensemble_predictions, axis=1)
ensemble_max = np.max(ensemble_predictions, axis=1)

plt.scatter(np.arange(0, n_test), tensor_data_test[1])
for i in range(n_test):
    plt.vlines(
        x=i, ymin=ensemble_min[i], ymax=ensemble_max[i], 
        colors='red', linewidth=2, alpha=0.7
    )
plt.show()
