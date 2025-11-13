# Test full model
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

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

from vae_attention.full_model import DeltaTimeAttentionVAE


# set number of cores to use
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
p_static = 3
p_interventions = 0
batch_size = 100

# custom W
np.random.seed(323425)
W = np.random.choice(
    [-1.5, -1, -0.8, -0.5, 1.5, 1, 0.8, 0.5],
    size=(k, p)
)
W[:, 0:p0] = 0

beta = np.array([-1, 1, -1, 1, -1])
beta = beta[..., None] * np.ones([k, n_timepoints])
beta[0, 1:] = [-2., -2., -1., -1]
beta[1, 1:] = [2., 3., 1., 1]
beta[2, 1:] = [0, 0, 0, 0]

beta_time = np.array([0, 1, 2, 1, 0, -1])
beta_static = np.random.choice([-1, 1], size=p_static)
beta_interventions = np.random.choice([-1, 1], size=(n_measurements, p_interventions))

dict_gen = data_generation.multi_longitudinal_data_generation(
    n, k, p, n_timepoints, n_measurements,
    p_static=p_static,
    p_interventions=p_interventions,
    noise_scale = 0.5,
    W=W,
    beta=beta,
    beta_time=beta_time,
    beta_interventions=beta_interventions,
    beta_static=beta_static,
    missing_prob=0.3
)

y = dict_gen["y"]
X = dict_gen["X"]
X_static = dict_gen["X_static"]
# X_interventions = dict_gen["X_interventions"]
Z = dict_gen["Z"]

# y0 (y at baseline) is actually an additional feature, because it is measured before any intervention
y.shape
y_baseline = y[:, :, 0:1]
y_baseline.shape
# the actual target is then y from t=1
y_target = y[:, :, 1:]
y_target.shape

n_timepoints = n_timepoints -1

# get tensors
X_tensor = torch.FloatTensor(X).to(torch.device("cpu"))
X_static_tensor = torch.FloatTensor(X_static).to(torch.device("cpu"))
y_target_tensor = torch.FloatTensor(y_target).to(torch.device("cpu"))
y_baseline_tensor = torch.FloatTensor(y_baseline).to(torch.device("cpu"))

# split data
data_split = data_loading_wrappers.DataSplit(X.shape[0], test_size=n_test, val_size=n_val)
print("train: ", len(data_split.train_index), 
    "val: ", len(data_split.val_index),
    "test: ", len(data_split.test_index)
)

tensor_data_train = data_split.get_train(X_tensor, X_static_tensor, y_baseline_tensor, y_target_tensor)
tensor_data_test = data_split.get_test(X_tensor, X_static_tensor, y_baseline_tensor, y_target_tensor)
tensor_data_val = data_split.get_val(X_tensor, X_static_tensor, y_baseline_tensor, y_target_tensor)

# make tensor data loaders
reshape = True
drop_missing = True

train_dataloader = data_loading_wrappers.make_data_loader(
    *tensor_data_train, batch_size=batch_size, feature_dimensions=-1, reshape=reshape, drop_missing=drop_missing
)
test_dataloader = data_loading_wrappers.make_data_loader(
    *tensor_data_test, batch_size=batch_size, feature_dimensions=-1, reshape=reshape, drop_missing=drop_missing
)
val_dataloader = data_loading_wrappers.make_data_loader(
    *tensor_data_val, batch_size=batch_size, feature_dimensions=-1, reshape=reshape, drop_missing=drop_missing
)


# 4. Training Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

latent_dim = k * 2
# for the number of heads
transformer_input_dim = 256
transformer_dim_feedforward = transformer_input_dim * 4


model = DeltaTimeAttentionVAE(
    input_dim=p,
    patient_features_dim=p_static,
    n_timepoints=n_timepoints,
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

count_parameters(model)


# Details
batch = next(iter(train_dataloader))
# ------------------ process input batch ------------------
x, y_baseline, patients_static_features = model.preprocess_input(batch)
x.shape
y_baseline.shape
patients_static_features.shape

# ---------------------------- VAE ----------------------------
x_hat, mu, logvar = model.vae(x)
x_hat.shape
mu.shape

# --------------------- Concat Static fatures ----------------------
h = torch.cat([x_hat, y_baseline], dim=-1)
h.shape

# ------ positional encoding and projection ------
h_exp = model.expand_input_in_time(h)
h_exp.shape

# --------------- Projection to transformer input dimension -----------
h_in = model.projection_to_transformer(h_exp)
h_in.shape

# -------- Generate FiLM parameters γ and β from static patient features --------
h_mod = model.film_generator(patients_static_features, h_in)
h_mod.shape

# --------------------- Time positional embedding ---------------------
h_time = model.pos_encoder(h_mod)
h_time.shape

# ----------------------- Transformer ------------------------------
h_out = model.transformer_module(h_time, attn_mask=model.causal_mask)
h_out.shape

# --------------------- Predict outcomes ---------------------
y_hat = model.outcome_prediction(h_out)
y_hat.shape

m_out = model(batch)
model.loss(m_out, batch)
m_out[1].shape
batch[3].shape

# ----------- Training Loop -----------
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

model.get_attention_weights(tensor_data_train)
