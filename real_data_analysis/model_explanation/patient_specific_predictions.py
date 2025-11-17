# Patient specific predictions and shap explanations
import pickle
import os
import copy
from pathlib import Path
import argparse

import shap
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.utils.convert_to_array import convert_to_static_multidim_array, convert_to_longitudinal_multidim_array
from src.utils.features_preprocessing import preprocess_train, preprocess_transform
from src.utils.config_reader import read_config
from src.utils.get_arrays import load_and_process_data
from src.utils.prepare_data_for_shap import prepare_data_for_shap
from src.utils import data_loading_wrappers
from src.utils.get_available_datapoints_indeces import get_indeces

# Script specific modules
# Must be in the same directory where model_fitting.py is run
from full_model import DeltaTimeAttentionVAE


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
FEATURES_KEYS = list(config_dict["preprocess"].keys())[:-1]

PATH_PLOTS = config_dict["script_parameters"]["patient_specific_plots_folder"]
os.makedirs(PATH_PLOTS, exist_ok = True)

# --------------------------------------------------------
# -------------------- Load data -------------------------
# --------------------------------------------------------
dict_arrays = load_and_process_data(config_dict, data_dir=PATH_DATA)
n_individuals = dict_arrays["genes"].shape[0]
p_gene = dict_arrays["genes"].shape[2]
p_metab = dict_arrays["metabolites"].shape[2]
p_static = dict_arrays["static_patient_features"].shape[2]
n_timepoints = dict_arrays["y_target"].shape[2]

print("\n preprocessing dict: ", config_dict["preprocess"])

where_all = get_indeces(dict_arrays)

genes_names = pd.read_csv(os.path.join(PATH_DATA, "genes_names.csv"), header=0, sep=";")
genes_names = genes_names["column_names"].to_numpy()
metab_names = pd.read_csv(os.path.join(PATH_DATA, "metab_names.csv"), header=0, sep=";")
metab_names = metab_names["column_names"].to_numpy()

all_features_names = np.concatenate([
    genes_names,
    metab_names,
    np.array(config_dict["data_arrays"]["static_patient_features"]),
    np.array(["Baseline"])
])

# Load pickle files
with open(os.path.join(PATH_RESULTS, "all_scalers"), "rb") as fp:   # Pickling scalers
    all_scalers = pickle.load(fp)

with open(os.path.join(PATH_RESULTS, "all_shap_values"), "rb") as fp:
    all_shap_values = pickle.load(fp)
print("Shap length: ", len(all_shap_values))

# Load torch models
all_models = []
for fold in range(N_FOLDS):
    print(f"Loading model fold {fold+1} of {N_FOLDS}")

    PATH = f"{PATH_RESULTS}/model_{fold}"
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
print("---------------------------- Running SHAP -------------------------")
dict_shap = {key: array for key, array in dict_arrays.items() if key in config_dict["data_arrays"].keys()}
features_combined, features_label_per_folds = prepare_data_for_shap(
    dict_shap,
    all_scalers,
    config_dict,
    verbose=False
)
print("\n Shape background_data for SHAP: ", features_combined[0].shape)

# -----------------------------------------------------------
# Get model predictions from all background data
all_predictions = []
all_ground_truth = []
for fold in range(N_FOLDS):
    
    y_true = features_label_per_folds[fold][-1]
    y_baseline = features_label_per_folds[fold][-2]
    all_ground_truth.append(
        np.concatenate([y_baseline, y_true], axis=-1)
    )

    tensor_input = features_label_per_folds[fold][:-1]
    model = all_models[fold]
    model.eval()
    with torch.no_grad():
        y_pred = model(tensor_input)[2].numpy()

    all_predictions.append(
        np.concatenate([y_baseline, y_pred], axis=-1)
    )

# -----------------------------------------------------------
# average the predictions over folds before plotting
target_predictions = np.array(all_predictions).mean(axis=0)
target_ground_truth = np.array(all_ground_truth).mean(axis=0)

slice_start = 0
slice_end = 0
colors_seq = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

for patient_id in range(n_individuals):
    # make folder for patient specific plots
    path_patient_plots = f"{PATH_PLOTS}/patient_{patient_id}"
    os.makedirs(path_patient_plots, exist_ok = True)

    patient_not_na = where_all[patient_id]
    sum_notna = patient_not_na.sum()
    slice_end += sum_notna

    # plot of true and predicted trajectories
    patient_pred = target_predictions[slice_start:slice_end]
    patient_ground_truth = target_ground_truth[slice_start:slice_end]

    fig = plt.figure()
    for meal in range(sum_notna):
        plt.plot(patient_ground_truth[meal], color=colors_seq[meal])
        plt.plot(patient_pred[meal], color=colors_seq[meal], linestyle="dashed")
    plt.xticks(range(0, n_timepoints + 1))
    plt.xlabel("Time")
    plt.title("Standardized-log TG")
    fig.savefig(f"{path_patient_plots}/predicted_y.pdf", format="pdf")
    plt.close()

    slice_start += sum_notna


# -----------------------------------------------------------------------
# plots in original scale

# inverse-transform
all_predictions_original = []
all_ground_truth_original = []
for fold in range(N_FOLDS):
    all_predictions_original.append(
        np.concatenate([
            all_scalers[fold]["y_baseline"][0].inverse_transform(all_predictions[fold][:, 0:1]),
            all_scalers[fold]["y_target"][0].inverse_transform(all_predictions[fold][:, 1:])
        ], axis=1)
    )
    all_ground_truth_original.append(
        np.concatenate([
            all_scalers[fold]["y_baseline"][0].inverse_transform(all_ground_truth[fold][:, 0:1]),
            all_scalers[fold]["y_target"][0].inverse_transform(all_ground_truth[fold][:, 1:])
        ], axis=1)
    )


all_predictions_original = np.exp(np.array(all_predictions_original))
all_ground_truth_original = np.exp(np.array(all_ground_truth_original))

target_predictions = all_predictions_original.mean(axis=0)
target_ground_truth = all_ground_truth_original.mean(axis=0)

# Calculate summary statistics and save
rmse = np.sqrt(np.square(target_predictions[..., 1:] - target_ground_truth[..., 1:]).mean())
print("RMSE predictions: ", rmse)
pd.DataFrame(np.array([rmse]), columns=["RMSE"])

slice_start = 0
slice_end = 0
colors_seq = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

for patient_id in range(n_individuals):
    # make folder for patient specific plots
    path_patient_plots = f"{PATH_PLOTS}/patient_{patient_id}"
    os.makedirs(path_patient_plots, exist_ok = True)

    patient_not_na = where_all[patient_id]
    sum_notna = patient_not_na.sum()
    slice_end += sum_notna

    # plot of true and predicted trajectories
    patient_pred = target_predictions[slice_start:slice_end]
    patient_ground_truth = target_ground_truth[slice_start:slice_end]

    fig = plt.figure()
    for meal in range(sum_notna):
        plt.plot(patient_ground_truth[meal], color=colors_seq[meal])
        plt.plot(patient_pred[meal], color=colors_seq[meal], linestyle="dashed")
    plt.xticks(range(0, n_timepoints + 1))
    plt.xlabel("Time")
    plt.title("Original scale TG")
    fig.savefig(f"{path_patient_plots}/original_scale_predicted_y.pdf", format="pdf")
    plt.close()
    
    slice_start += sum_notna

print("\n ---------------- END ------------------")
