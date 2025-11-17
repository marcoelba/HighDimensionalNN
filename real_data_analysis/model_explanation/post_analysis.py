# analysis of loss components
import os
import pickle
import copy

import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from real_data_analysis.utils.convert_to_array import convert_to_static_multidim_array, convert_to_longitudinal_multidim_array
from real_data_analysis.utils.features_preprocessing import preprocess_train, preprocess_transform

from real_data_analysis.model_genes_metabolomics_no_vae.get_arrays import load_and_process_data
from real_data_analysis.model_genes_metabolomics_no_vae.config_reader import read_config
from real_data_analysis.model_genes_metabolomics_no_vae.full_model import DeltaTimeAttentionVAE

from src.utils import data_loading_wrappers


def loss_components(m_out, batch, model):
    """
    Loss function. The structure depends on the batch data.
    To be modified according to the data used.

    VAE loss + Prediction Loss (here MSE)
    """
    x_genes, x_metab, y_baseline, patients_static_features = model.process_batch(batch)
    # vae output: x_hat, mu, logvar, z_hat
    # Reconstruction loss (MSE)
    genes_vae_loss = nn.functional.mse_loss(m_out[0][0], x_genes, reduction='none')
    metab_vae_loss = nn.functional.mse_loss(m_out[1][0], x_metab, reduction='none')
    # KL divergence
    genes_KLD = -0.5 * torch.sum(1 + m_out[0][2] - m_out[0][1].pow(2) - m_out[0][2].exp())
    metab_KLD = -0.5 * torch.sum(1 + m_out[1][2] - m_out[1][1].pow(2) - m_out[1][2].exp())
    # label prediction loss
    PredMSE = nn.functional.mse_loss(m_out[-1], batch[-1], reduction='none')

    out_dict = dict(
        genes_vae_loss=genes_vae_loss,
        metab_vae_loss=metab_vae_loss,
        genes_KLD=genes_KLD,
        metab_KLD=metab_KLD,
        PredMSE=PredMSE
    )

    return out_dict


# Create directory for results
PATH_MODELS = "./real_data_analysis/results/res_train_v4_no_vae"

config_dict = read_config("./real_data_analysis/model_genes_metabolomics_no_vae/config.ini")
DEVICE = torch.device(config_dict["training_parameters"]["device"])
N_FOLDS = config_dict["training_parameters"]["n_folds"]


# --------------------------------------------------------
# -------------------- Load data -------------------------
# --------------------------------------------------------
dict_arrays = load_and_process_data(config_dict, data_dir="./real_data_analysis/data")
n_individuals = dict_arrays["genes"].shape[0]
p_gene = dict_arrays["genes"].shape[2]
p_metab = dict_arrays["metabolites"].shape[2]
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
        input_dim_genes=p_gene,
        input_dim_metab=p_metab,
        input_patient_features_dim=p_static,
        n_timepoints=n_timepoints,
        model_config=config_dict["model_params"]
    ).to(DEVICE)
    model.load_state_dict(torch.load(PATH))
    all_models.append(model)


# LOSS components
mse_all_folds = []
genes_loss_all_folds = []
metab_loss_all_folds = []
genes_kl_loss = []
metab_kl_loss = []

for fold in range(N_FOLDS):
    # apply feature preprocessing
    dict_arrays_preproc = preprocess_transform(
        copy.deepcopy(dict_arrays), all_scalers[fold], config_dict
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
            batch=tensor_input,
            model=model
        )
    mse_all_folds.append(loss["PredMSE"].numpy())
    genes_loss_all_folds.append(loss["genes_vae_loss"].numpy())
    metab_loss_all_folds.append(loss["metab_vae_loss"].numpy())
    genes_kl_loss.append(loss["genes_KLD"].numpy())
    metab_kl_loss.append(loss["metab_KLD"].numpy())

pred_mse = np.array(mse_all_folds).mean(axis=0)
genes_loss = np.array(genes_loss_all_folds).mean(axis=0)
metab_loss = np.array(metab_loss_all_folds).mean(axis=0)
genes_kl = np.array(genes_kl_loss).mean(axis=0)
metab_kl = np.array(metab_kl_loss).mean(axis=0)

print("\n Average Prediction MSE: ", pred_mse.mean(axis=0))
print("\n Average Genes Reconstruction MSE: ", genes_loss.mean())
print("\n Average Metabolites Reconstruction MSE: ", metab_loss.mean())
print("\n Average genes KL divergence: ", genes_kl)
print("\n Average metabolites KL divergence: ", metab_kl)
