# analysis of loss components
import os
import pickle
import copy

import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from src.vae_attention.full_model import DeltaTimeAttentionVAE
from real_data_analysis.convert_to_array import convert_to_static_multidim_array, convert_to_longitudinal_multidim_array
from real_data_analysis.features_preprocessing import preprocess, preprocess_transform
from src.utils import data_loading_wrappers
from real_data_analysis.config_reader import read_config
from real_data_analysis.get_arrays import load_and_process_data


def loss_components(m_out, batch):
    """
    Loss function. The structure depends on the batch data.
    To be modified according to the data used.

    VAE loss + Prediction Loss (here MSE)
    """
    # Reconstruction loss (MSE)
    BCE = nn.functional.mse_loss(m_out[0], batch[0], reduction='none')
    # KL divergence
    KLD = -0.5 * torch.sum(1 + m_out[3] - m_out[2].pow(2) - m_out[3].exp())
    # label prediction loss
    PredMSE = nn.functional.mse_loss(m_out[1], batch[3], reduction='none')

    return dict(BCE=BCE, KLD=KLD, PredMSE=PredMSE)


# Create directory for results
PATH_MODELS = "./res_train"

config_dict = read_config("./config.ini")
DEVICE = torch.device(config_dict["training_parameters"]["device"])
N_FOLDS = config_dict["training_parameters"]["n_folds"]


# --------------------------------------------------------
# -------------------- Load data -------------------------
# --------------------------------------------------------
dict_arrays, features_to_preprocess = load_and_process_data(config_dict, data_dir="./")
n_individuals = dict_arrays["genes"].shape[0]
p = dict_arrays["genes"].shape[2]
p_static = dict_arrays["static_patient_features"].shape[2]
n_timepoints = dict_arrays["y_target"].shape[2]

# Load pickle files
with open(f"{PATH_MODELS}/all_scalers", "rb") as fp:   # Pickling scalers
    all_scalers = pickle.load(fp)


for fold in range(N_FOLDS):
    print(f"\n -------- FOLD: {fold} --------------")
    scalers = all_scalers[fold]
    all_mean = []
    all_var = []
    for kk, dd in scalers.items():
        for k, s in dd.items():
            all_mean.append(s.mean_)
            all_var.append(s.var_)
    print("\n min/max of mean: ", np.array(all_mean).min(), np.array(all_mean).max())
    print("\n min/max of var: ", np.array(all_var).min(), np.array(all_var).max())


all_models = []
for fold in range(N_FOLDS):
    print(f"Loading model fold {fold+1} of {N_FOLDS}")

    PATH = f"{PATH_MODELS}/model_{fold}"
    model = DeltaTimeAttentionVAE(
        input_dim=p,
        patient_features_dim=p_static,
        n_timepoints=n_timepoints,
        vae_latent_dim=config_dict["model_params"]["latent_dim"],
        vae_input_to_latent_dim=64,
        max_len_position_enc=10,
        transformer_input_dim=config_dict["model_params"]["transformer_input_dim"],
        transformer_dim_feedforward=config_dict["model_params"]["transformer_dim_feedforward"],
        nheads=config_dict["model_params"]["n_heads"],
        dropout=0.1,
        dropout_attention=0.1,
        prediction_weight=1.0
    ).to(DEVICE)
    model.load_state_dict(torch.load(PATH))
    all_models.append(model)


# LOSS components
mse_all_folds = []
bce_all_folds = []
attn_weights = []

for fold in range(N_FOLDS):
    # apply feature preprocessing
    dict_arrays_preproc = preprocess_transform(
        copy.deepcopy(dict_arrays), all_scalers[fold], features_to_preprocess
    )
    # remove last dimension for outcome with only one dimension
    if dict_arrays_preproc["y_target"].shape[-1] == 1:
        dict_arrays_preproc["y_target"] = dict_arrays_preproc["y_target"][..., 0]
        dict_arrays_preproc["y_baseline"] = dict_arrays_preproc["y_baseline"][..., 0]

    # keep only input features and get tensors
    tensor_data = [
        torch.FloatTensor(array).to(DEVICE) for key, array in dict_arrays_preproc.items()
    ]
    # make longitudinal
    tensor_dataset = data_loading_wrappers.CustomDataset(
        *tensor_data,
        reshape=True,
        remove_missing=True,
        feature_dimensions=-1,
        device=DEVICE
    )
    tensor_input = tensor_dataset.arrays
    # fold-model prediction
    model = all_models[fold]
    model.eval()
    with torch.no_grad():
        pred = model(tensor_input)
        # x_hat, y_hat, mu, logvar
        loss = loss_components(
            m_out=pred,
            batch=tensor_input
        )
        mse_all_folds.append(loss["PredMSE"].numpy())
        bce_all_folds.append(loss["BCE"].numpy())

    attn_weights.append(model.get_attention_weights(tensor_input))


print("Average Prediction MSE: ", np.array(mse_all_folds).mean())
print("Average Reconstruction MSE: ", np.array(bce_all_folds).mean())

pred_mse = np.array(mse_all_folds).mean(axis=0)
pred_mse_x = np.array(bce_all_folds).mean(axis=0)
attn_weights = np.array(attn_weights).mean(axis=0)

# save attention weights
with open(f"{PATH_MODELS}/attn_weights", "wb") as fp:
    pickle.dump(attn_weights, fp)
