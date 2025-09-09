# Test full model
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
from utils import data_generation

from vae_attention.modules.transformer import TransformerEncoderLayerWithWeights
from vae_attention.modules.sinusoidal_position_encoder import SinusoidalPositionalEncoding
from vae_attention.modules.vae import VAE


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
batch_size = 50

# custom W
np.random.seed(323425)
W = np.random.choice(
    [-1.5, -1, -0.8, -0.5, 1.5, 1, 0.8, 0.5],
    size=(k, p)
)
first_half = range(0, int(p / 2))
second_half = range(int(p / 2), p)
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


# y0 (y at baseline) is actually an additional feature, because it is measured before any intervention
y.shape
y_baseline = y[:, :, 0:1]
y_baseline.shape
# the actual target is then y from t=1
y_target = y[:, :, 1:]
y_target.shape

# get tensors
X_tensor = torch.FloatTensor(X).to(torch.device("cpu"))
Z_tensor = torch.FloatTensor(Z).to(torch.device("cpu"))
y_target_tensor = torch.FloatTensor(y_target).to(torch.device("cpu"))
y_baseline_tensor = torch.FloatTensor(y_baseline).to(torch.device("cpu"))

# split data
data_split = data_loading_wrappers.DataSplit(X.shape[0], test_size=n_test, val_size=n_val)
print("train: ", len(data_split.train_index), 
    "val: ", len(data_split.val_index),
    "test: ", len(data_split.test_index)
)

data_train = data_split.get_train(X, y_baseline, Z, y_target)
data_test = data_split.get_test(X, y_baseline, Z, y_target)
data_val = data_split.get_val(X, y_baseline, Z, y_target)

tensor_data_train = data_split.get_train(X_tensor, y_baseline_tensor, y_target_tensor)
tensor_data_test = data_split.get_test(X_tensor, y_baseline_tensor, y_target_tensor)
tensor_data_val = data_split.get_val(X_tensor, y_baseline_tensor, y_target_tensor)

# make tensor data loaders
train_dataloader = data_loading_wrappers.make_data_loader(*tensor_data_train, batch_size=batch_size)
test_dataloader = data_loading_wrappers.make_data_loader(*tensor_data_test, batch_size=batch_size)
val_dataloader = data_loading_wrappers.make_data_loader(*tensor_data_val, batch_size=batch_size)


# 4. Training Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

latent_dim = k * 2
# for the number of heads
transformer_input_dim = 256
transformer_dim_feedforward = transformer_input_dim * 4


model = DeltaTimeAttentionVAE(
    input_dim=p,
    n_timepoints=n_timepoints-1,
    vae_latent_dim=latent_dim,
    vae_input_to_latent_dim=64,
    max_len_position_enc=10,
    transformer_input_dim=transformer_input_dim,
    transformer_dim_feedforward=transformer_dim_feedforward,
    nheads=4,
    dropout=0.1,
    dropout_attention=0.1,
    prediction_weight=1.0
).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


batch = next(iter(train_dataloader))

# ------------------ process input batch ------------------
x, y_baseline = model.preprocess_input(batch)
x.shape
y_baseline.shape

# ---------------------------- VAE ----------------------------
# x_flat = x.view(-1, self.input_dim)  # (batch_size * max_measurements, input_dim)
x_hat, mu, logvar = model.vae(x)
x_hat.shape
mu.shape
logvar.shape

# ------ concatenate with y0, positional encoding and projection ------
h_time = model.make_transformer_input(x_hat, y_baseline)
h_time.shape

# ----------------------- Transformer ------------------------------
h_out = model.transformer_module(h_time, attn_mask=model.causal_mask)
h_out.shape

# --------------------- Predict outcomes ---------------------
y_hat = model.outcome_prediction(h_out)
y_hat.shape

# check attention weights
attn_weights = model.get_attention_weights(tensor_data_train)
attn_weights.shape
