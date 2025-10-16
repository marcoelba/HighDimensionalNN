# SHAP explanations
import pickle
import os
import copy

import shap
import torch
import pandas as pd
import numpy as np

from real_data_analysis.utils.convert_to_array import convert_to_static_multidim_array, convert_to_longitudinal_multidim_array
from real_data_analysis.utils.features_preprocessing import preprocess_train, preprocess_transform

from real_data_analysis.model_genes_metabolomics.get_arrays import load_and_process_data
from real_data_analysis.model_genes_metabolomics.config_reader import read_config
from real_data_analysis.model_genes_metabolomics.full_model import DeltaTimeAttentionVAE

from src.utils import data_loading_wrappers


# Read config
PATH_MODELS = "./real_data_analysis/results/res_train_v3"

config_dict = read_config("./real_data_analysis/model_genes_metabolomics/config.ini")
DEVICE = torch.device(config_dict["training_parameters"]["device"])
N_FOLDS = config_dict["training_parameters"]["n_folds"]
FEATURES_KEYS = list(config_dict["preprocess"].keys())[:-1]

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

# Load torch models
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


# Define a Torch ensemble model that takes in input a list of models
class EnsembleModel(torch.nn.Module):
    def __init__(self, model_list, time_to_explain):
        super(EnsembleModel, self).__init__()
        self.models = torch.nn.ModuleList(model_list)
        self.time_to_explain = time_to_explain

    def forward(self, *x):
        x_list = list(x)
        all_outputs = []
        for model in self.models:
            model.eval()
            output = model(x_list)[2][:, [self.time_to_explain]]
            all_outputs.append(output)
        return torch.stack(all_outputs).mean(dim=0)

# -------------------------------------------------------------------------
# ---------------------- Run SHAP explanation -----------------------------
# -------------------------------------------------------------------------
print("---------------- Running SHAP ---------------")
dict_shap = {key: array for key, array in dict_arrays.items() if key in FEATURES_KEYS}
background_data = prepare_data_for_shap(dict_shap, subsample=False)
print("Shape background_data for SHAP: ", background_data[0].shape)
explain_data = prepare_data_for_shap(dict_shap, subsample=False)
print("Shape explain data for SHAP: ", explain_data[0].shape)

all_shap_values = []
for time_point in range(n_timepoints):
    ensemble_model = EnsembleModel(all_models, time_to_explain=time_point)
    explainer = shap.GradientExplainer(ensemble_model, background_data)
    shap_values = explainer.shap_values(explain_data)
    all_shap_values.append(shap_values)

# save shap values to pickle
with open("all_shap_values", "wb") as fp:
    pickle.dump(all_shap_values, fp)
