# analysis of loss components
import os
import pickle
import copy
from pathlib import Path
import argparse

import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from src.utils.convert_to_array import convert_to_static_multidim_array, convert_to_longitudinal_multidim_array
from src.utils.features_preprocessing import preprocess_train, preprocess_transform
from src.utils.config_reader import read_config
from src.utils.get_arrays import load_and_process_data
from src.utils import data_loading_wrappers

# Script specific modules
# Must be in the same directory where model_fitting.py is run
from full_model import Model


# read input arguments from console
parser = argparse.ArgumentParser(description='Run program with custom config and modules')
parser.add_argument('-c', '--config', required=True, help='Path to config.ini file')
args = parser.parse_args()

# Load config file
config_path = Path(args.config)
if not config_path.exists():
    print(f"Error: Config file not found: {config_path}")
    sys.exit(1)
config_dict = read_config(config_path)

PATH_RESULTS = config_dict["script_parameters"]["results_folder"]
PATH_DATA = config_dict["script_parameters"]["data_folder"]
DEVICE = torch.device(config_dict["training_parameters"]["device"])
N_FOLDS = config_dict["training_parameters"]["n_folds"]

# --------------------------------------------------------
# -------------------- Load data -------------------------
# --------------------------------------------------------
dict_arrays = load_and_process_data(config_dict, data_dir=PATH_DATA)
n_individuals = dict_arrays["genes"].shape[0]
p_gene = dict_arrays["genes"].shape[2]
p_metab = dict_arrays["metabolites"].shape[2]
p_static = dict_arrays["static_patient_features"].shape[2]
n_timepoints = dict_arrays["y_target"].shape[2]

# Load pickle files
with open(f"{PATH_RESULTS}/all_scalers", "rb") as fp:   # Pickling scalers
    all_scalers = pickle.load(fp)

all_models = []
for fold in range(N_FOLDS):
    print(f"Loading model fold {fold+1} of {N_FOLDS}")

    PATH = f"{PATH_RESULTS}/model_{fold}"
    model = Model(
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
        loss = model.loss(
            m_out=pred,
            batch=tensor_input,
            reduction='mean'
        )
    mse_all_folds.append(loss[0].numpy())
    # genes_loss_all_folds.append(loss["genes_vae_loss"].numpy())
    # metab_loss_all_folds.append(loss["metab_vae_loss"].numpy())
    # genes_kl_loss.append(loss["genes_KLD"].numpy())
    # metab_kl_loss.append(loss["metab_KLD"].numpy())

pred_mse = np.array(mse_all_folds).mean()
# genes_loss = np.array(genes_loss_all_folds).mean(axis=0)
# metab_loss = np.array(metab_loss_all_folds).mean(axis=0)
# genes_kl = np.array(genes_kl_loss).mean(axis=0)
# metab_kl = np.array(metab_kl_loss).mean(axis=0)

print("\n Average Prediction MSE: ", pred_mse)
# print("\n Average Genes Reconstruction MSE: ", genes_loss.mean())
# print("\n Average Metabolites Reconstruction MSE: ", metab_loss.mean())
# print("\n Average genes KL divergence: ", genes_kl)
# print("\n Average metabolites KL divergence: ", metab_kl)
