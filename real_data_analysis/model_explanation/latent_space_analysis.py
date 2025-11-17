# VAE analysis
import os
import pickle
import copy

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

from src.vae_attention.full_model import DeltaTimeAttentionVAE
from real_data_analysis.convert_to_array import convert_to_static_multidim_array, convert_to_longitudinal_multidim_array
from real_data_analysis.features_preprocessing import preprocess, preprocess_transform
from src.utils import data_loading_wrappers
from real_data_analysis.config_reader import read_config
from real_data_analysis.get_arrays import load_and_process_data


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
all_correlations = []

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
    cor_x = []
    with torch.no_grad():
        X_hat = model(tensor_input)[0].cpu().numpy()
        for cov in range(X_hat.shape[1]):
            cor_x.append(pearsonr(X_hat[:, cov], tensor_input[0][:, cov])[0])
    all_correlations.append(cor_x)

correlations = np.array(all_correlations).mean(axis=0)
print("First 5 correlations: ", np.round(correlations[0:5], 3))

# correlations
with open(f"correlations", "wb") as fp:
    pickle.dump(correlations, fp)

# Latent space analysis through the decoder and
# Latent space perturbation
# Start from a baseline input for ONE observation
perturbation_range = np.linspace(-2, 2, 5) # perturb from -2 to 2 std dev
all_folds_perturbations = []

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
    # fold-model
    x = tensor_input[0]
    model = all_models[fold]
    model.eval()
    with torch.no_grad():
        # get the estimated latent space first
        mu, logvar = model.vae.encode(x)
        z_hat = model.vae.reparameterize(mu, logvar)
        x_hat = model.vae.decode(z_hat)

    # Choose a latent dimension to perturb and a perturbation amount
    list_sensitivities = []

    for latent_dim_to_perturb in range(config_dict["model_params"]["latent_dim"]):
        reconstruction_changes = [] # List to store change per feature

        for eps in perturbation_range:
            z_perturbed = z_hat.clone()
            z_perturbed[0, latent_dim_to_perturb] += eps
            with torch.no_grad():
                pert_x_hat = model.vae.decode(z_perturbed)
                # take difference
                delta = pert_x_hat - x_hat
                reconstruction_changes.append(delta.numpy())

        # one element of reconstruction_changes is a matrix [n_perturbations, p_features]
        reconstruction_changes = np.array(reconstruction_changes)

        # For each feature, calculate its sensitivity to the latent dimension
        # (e.g., variance across perturbations or max absolute change)
        feature_sensitivity = np.var(reconstruction_changes, axis=0)
        # feature_sensitivity.shape
        average_feature_sensitivity = feature_sensitivity.mean(axis=0)
        # feature_sensitivity = np.max(np.abs(reconstruction_changes), axis=0)
        list_sensitivities.append(average_feature_sensitivity)
    
    # make one array (one row per latent dim)
    array_sensitivities = np.array(list_sensitivities)
    all_folds_perturbations.append(array_sensitivities)

array_folds_perturbations = np.array(all_folds_perturbations)
with open(f"array_folds_perturbations", "wb") as fp:
    pickle.dump(array_folds_perturbations, fp)

top_sensitive_features = np.argsort(array_folds_perturbations.mean(axis=0), axis=1)[:, 0:5]
latent_dim = 0
print(f"Features most sensitive to latent dim {latent_dim}: \n {top_sensitive_features[latent_dim, :]}")
