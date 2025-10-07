# SHAP explanations
import pickle
import os
import copy

import shap
import torch
import pandas as pd
import numpy as np

from real_data_analysis.model_use_vae_z.full_model import DeltaTimeAttentionVAE
from real_data_analysis.convert_to_array import convert_to_static_multidim_array, convert_to_longitudinal_multidim_array
from real_data_analysis.features_preprocessing import preprocess, preprocess_transform
from src.utils import data_loading_wrappers
from real_data_analysis.config_reader import read_config
from real_data_analysis.get_arrays import load_and_process_data


# Read config
PATH_MODELS = "./res_train_v2"

config_dict = read_config("config.ini")
DEVICE = torch.device(config_dict["training_parameters"]["device"])
N_FOLDS = config_dict["training_parameters"]["n_folds"]
FEATURES_KEYS = list(config_dict["preprocess"].keys())[:-1]

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

# Load torch models
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
            output = model(x_list)[1][:, [self.time_to_explain]]
            all_outputs.append(output)
        return torch.stack(all_outputs).mean(dim=0)

# pre-process the input data with all folds scalers at once
def prepare_data_for_shap(dict_shap, subsample=False, n_background=100, patient_index=None):
    tensor_input_per_fold = []

    for fold in range(N_FOLDS):
        # apply feature preprocessing
        dict_arrays_preproc = preprocess_transform(
            copy.deepcopy(dict_shap), all_scalers[fold], features_to_preprocess
        )
        if dict_arrays_preproc["y_baseline"].shape[-1] == 1:
            dict_arrays_preproc["y_baseline"] = dict_arrays_preproc["y_baseline"][..., 0]

        if patient_index is not None:
            tensor_data = [
                torch.FloatTensor(array[patient_index]).to(DEVICE) for key, array in dict_arrays_preproc.items()
            ]
        elif (patient_index is None) and subsample:
            index = np.random.choice(next(iter(dict_shap.values())).shape[0], n_background, replace=False)
            tensor_data = [
                torch.FloatTensor(array[index]).to(DEVICE) for key, array in dict_arrays_preproc.items()
            ]
        else:
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
        # append
        tensor_input_per_fold.append(tensor_input)
    
    combined_tensors_per_fold = []
    for feature in range(len(FEATURES_KEYS)):
        tensor_feature = []
        for fold in range(N_FOLDS):
            tensor_feature.append(tensor_input_per_fold[fold][feature])
        combined_tensors_per_fold.append(torch.cat(tensor_feature, dim=0))
    
    return combined_tensors_per_fold


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

# # save shap values ot pickle
# with open("all_shap_values", "wb") as fp:
#     pickle.dump(all_shap_values, fp)
