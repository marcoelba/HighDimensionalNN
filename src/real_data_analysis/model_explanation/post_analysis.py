# analysis of loss components
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from src.utils.features_preprocessing import preprocess_train, preprocess_transform
from src.utils.config_reader import get_config
from src.utils.get_arrays import CustomData
from src.utils import data_loading_wrappers

# Script specific modules
# Must be in the same directory where model_fitting.py is run
from full_model import Model


# get config from console input arguments
config_dict = get_config()

PATH_RESULTS = config_dict["script_parameters"]["results_folder"]
DEVICE = torch.device(config_dict["training_parameters"]["device"])
N_FOLDS = config_dict["training_parameters"]["n_folds"]

# -------------------- Load data -------------------------
data = CustomData(config_dict, data_dir=config_dict["script_parameters"]["data_folder"])
dict_arrays = data.load_and_process_data(data_dir=config_dict["script_parameters"]["data_folder"])

# load model pipeline data
pipeline_data = EnsemblePipelineData(config_dict=config_dict)
all_scalers = pipeline_data.load_scalers()
model_paramerers = pipeline_data.load_model_paramerers()

# LOSS components
mse_all_folds = []
genes_loss_all_folds = []
metab_loss_all_folds = []
genes_kl_loss = []
metab_kl_loss = []

for fold in range(N_FOLDS):
    print(f"Loading model fold {fold+1} of {N_FOLDS}")
    path = f"{PATH_RESULTS}/model_{fold}"
    model = Model(
        input_dim_genes=model_paramerers["p_gene"],
        input_dim_metab=model_paramerers["p_metab"],
        input_patient_features_dim=model_paramerers["p_static"],
        n_timepoints=model_paramerers["n_timepoints"],
        model_config=config_dict["model_params"]
    ).to(DEVICE)
    model.load_state_dict(torch.load(path))

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
