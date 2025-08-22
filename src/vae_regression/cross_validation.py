# cross - validation
import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
from scipy.stats import pearsonr

import matplotlib.pyplot as plt
import shap

os.chdir("./src")

from vae_regression import data_generation
from vae_regression import training
from vae_regression import model_time_vae
from model_utils import utils

torch.get_num_threads()
torch.set_num_threads(6)


# generate data assuming an underlying latent space of dimension k
k = 5
n_train = 100
n_test = 200
n_val = 200
n = n_train + n_val + n_test
p = 600
p1 = 30
p0 = p - p1
n_timepoints = 5
n_measurements = 4
batch_size = 33

# custom W
np.random.seed(323425)
W = np.random.choice(
    [-1.5, -1, -0.8, -0.5, 1.5, 1, 0.8, 0.5],
    size=(k, p)
)
first_half = range(0, int(p/2))
second_half = range(int(p/2), p)
# # block structure
# W[0, first_half] = 0.0
# W[1, first_half] = 0.0
# W[3, second_half] = 0.0
# W[4, second_half] = 0.0
# # first p0 features do NOT have any effect
W[:, 0:p0] = 0

beta = np.array([-1, 1, -1, 1, -1])
beta = beta[..., None] * np.ones([k, n_timepoints])
beta[0, 1:] = [-2., -2., -1., -1]
beta[1, 1:] = [2., 3., 1., 1]
beta[2, 1:] = [0, 0, 0, 0]

beta_time = np.array([0, 1, 2, 1, 0, -1])

y, X, Z, beta = data_generation.multi_longitudinal_data_generation(
    n, k, p, n_timepoints, n_measurements,
    noise_scale = 0.5,
    W=W,
    beta=beta,
    beta_time=beta_time
)

# one measurement
plt.plot(y[0:5, 0, :].transpose())
plt.show()

# one patient
plt.plot(y[0, :, :].transpose())
plt.show()

# get tensors
X_tensor = torch.FloatTensor(X).to(torch.device("cpu"))
Z_tensor = torch.FloatTensor(Z).to(torch.device("cpu"))
y_tensor = torch.FloatTensor(y).to(torch.device("cpu"))


# Run a cross-validation loop
data_split = utils.DataSplit(X.shape[0], test_size=n_test, val_size=n_val)
# split data
print("train: ", len(data_split.train_index),
    "val: ", len(data_split.val_index),
    "test: ", len(data_split.test_index)
)

data_train = data_split.get_train(X, Z, y)
data_test = data_split.get_test(X, Z, y)

tensor_data_train = data_split.get_train(X_tensor, y_tensor)
tensor_data_test = data_split.get_test(X_tensor, y_tensor)

test_dataloader = utils.make_data_loader(*tensor_data_test, batch_size=batch_size)

num_epochs = 500

all_train_losses = []
all_val_losses = []
all_predictions = []
all_true = []

for observation in range(5):
    print(f"Running cross-validation {observation+1} of {n_train}")

    train_mask = torch.ones(n_train, dtype=torch.bool)
    train_mask[observation] = False

    # make train and validation data loader for the LOO cross-validation
    tensor_train_loo = [
        tensor_data_train[0][train_mask],
        tensor_data_train[1][train_mask]
    ]
    tensor_val_loo = [
        tensor_data_train[0][~train_mask],
        tensor_data_train[1][~train_mask],
    ]
    train_dataloader = utils.make_data_loader(*tensor_train_loo, batch_size=batch_size)
    val_dataloader = utils.make_data_loader(*tensor_val_loo, batch_size=1)

    # 4. Training Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    latent_dim = k * 2

    model = model_time_vae.TimeAwareRegVAE(
        input_dim=p,
        latent_dim=latent_dim,
        n_timepoints=n_timepoints,
        n_measurements=n_measurements,
        input_to_latent_dim=64,
        transformer_dim_feedforward=608,
        nhead=4,
        time_emb_dim=8,
        dropout_sigma=0.0,
        beta_vae=1.0,
        reconstruction_weight=1.0,
        prediction_weight=2.0
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 5. Training Loop
    trainer = training.Training(train_dataloader, val_dataloader)
    trainer.training_loop(model, optimizer, num_epochs)

    # Validate
    model.eval()
    with torch.no_grad():
        pred = model(tensor_val_loo[0])
        all_predictions.append(pred[1].numpy())
        all_true.append(tensor_val_loo[1].numpy())
    
    # store
    all_train_losses.append(trainer.losses["train"])
    all_val_losses.append(trainer.losses["val"])


predictions = np.concatenate(all_predictions, axis=0)
predictions.shape
ground_truth = np.concatenate(all_true, axis=0)
ground_truth.shape


all_mse = torch.nn.functional.mse_loss(torch.tensor(predictions), torch.tensor(ground_truth), reduction='none')
average_mse_per_time = torch.mean(all_mse, axis=[0, 1])

colors = plt.cm.Pastel1.colors

unit = 1
for measurement in range(n_measurements):
    plt.plot(predictions[unit][measurement, :], color=colors[measurement])
    plt.plot(ground_truth[unit][measurement, :],  linestyle='dashed', color=colors[measurement])
    plt.legend()
plt.show()


# plot
train_batch = all_train_losses[0]
val_batch = all_val_losses[0]

plt.plot(train_batch, label="train")
plt.plot(val_batch, label="val")
plt.vlines(np.argmin(val_batch), 0, max(val_batch), color="red")
plt.vlines(np.argmin(train_batch), 0, max(train_batch), color="blue")
plt.hlines(np.min(val_batch), 0, len(val_batch), color="red", linestyles="--")
plt.hlines(np.min(train_batch), 0, len(train_batch), color="blue", linestyles="--")
plt.legend()
plt.show()
