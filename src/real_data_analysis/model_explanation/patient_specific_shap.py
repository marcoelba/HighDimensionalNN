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
with open(f"{PATH_RESULTS}/all_scalers", "rb") as fp:   # Pickling scalers
    all_scalers = pickle.load(fp)

with open(f"{PATH_RESULTS}/all_shap_values", "rb") as fp:
    all_shap_values = pickle.load(fp)
print("Shap length: ", len(all_shap_values))

# Load torch models
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


class TorchScaler:
    def __init__(self, scaler, dtype=torch.float32):
        self.mean = torch.tensor(scaler.mean_, dtype=dtype)
        self.scale = torch.tensor(scaler.scale_, dtype=dtype)

    def transform(self, x):
        """
            x: torch.Tensor
        """
        return (x - self.mean) / self.scale

    def inverse_transform(self, x):
        """
            x: torch.Tensor
        """
        return x * self.scale + self.mean


# Define a Torch ensemble model that takes in input a list of models
class EnsembleModel(torch.nn.Module):
    def __init__(self, model_list, time_to_explain, torch_scalers_outcome=None):
        super(EnsembleModel, self).__init__()
        self.models = torch.nn.ModuleList(model_list)
        self.time_to_explain = time_to_explain
        self.torch_scalers_outcome = torch_scalers_outcome

    def forward(self, *x):
        x_list = list(x)
        all_outputs = []
        for fold, model in enumerate(self.models):
            model.eval()
            output = model(x_list)[2][:, [self.time_to_explain]]
            # transform output back to original scale
            if self.torch_scalers_outcome is not None:
                output = self.torch_scalers_outcome[fold].inverse_transform(output)
                output = torch.exp(output)
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
    config_dict
)
print("Shape background_data for SHAP: ", features_combined[0].shape)

torch_scalers_outcome = [TorchScaler(all_scalers[fold]['y_target'][0]) for fold in range(N_FOLDS)]

# Plot shap waterfall values
for time_point in range(n_timepoints):

    # Get base values for the explainer - time specific
    ensemble_model = EnsembleModel(all_models, time_to_explain=time_point, torch_scalers_outcome=torch_scalers_outcome)
    ensemble_model.eval()
    with torch.no_grad():
        predictions = ensemble_model(*features_combined)
        base_values = predictions.numpy()
    mean_base_value = np.array(np.array_split(base_values, N_FOLDS, axis=0)).mean()

    genes_shap = all_shap_values[time_point][0][..., -1]  # genes shap
    metab_shap = all_shap_values[time_point][1][..., -1]  # metabolites shap
    patient_clin_shap = all_shap_values[time_point][2][..., -1]  # patient shap
    baseline_shap = all_shap_values[time_point][3][..., -1]  # patient shap

    # Takes averages
    mean_genes_shap = np.array(np.array_split(genes_shap, N_FOLDS, axis=0)).mean(axis=0)
    mean_metab_shap = np.array(np.array_split(metab_shap, N_FOLDS, axis=0)).mean(axis=0)
    mean_patient_clin_shap = np.array(np.array_split(patient_clin_shap, N_FOLDS, axis=0)).mean(axis=0)
    mean_baseline_shap = np.array(np.array_split(baseline_shap, N_FOLDS, axis=0)).mean(axis=0)

    # data
    genes_data = np.array(np.array_split(features_combined[0], N_FOLDS, axis=0)).mean(axis=0)
    metab_data = np.array(np.array_split(features_combined[1], N_FOLDS, axis=0)).mean(axis=0)
    patient_clin_data = np.array(np.array_split(features_combined[2], N_FOLDS, axis=0)).mean(axis=0)
    baseline_data = np.array(np.array_split(features_combined[3], N_FOLDS, axis=0)).mean(axis=0)

    # make one array for shap and data
    mean_shap = np.concatenate([mean_genes_shap, mean_metab_shap, mean_patient_clin_shap, mean_baseline_shap], axis=-1)
    mean_data = np.concatenate([genes_data, metab_data, patient_clin_data, baseline_data], axis=-1)

    # slice over time
    slice_start = 0
    slice_end = 0
    for patient_id in range(n_individuals):
        # make folder for patient specific plots
        path_patient_plots = f"{PATH_PLOTS}/patient_{patient_id}"
        os.makedirs(path_patient_plots, exist_ok = True)

        patient_not_na = where_all[patient_id]
        sum_notna = patient_not_na.sum()
        slice_end += sum_notna

        mean_patient_data = np.concatenate([
            dict_arrays["genes"][patient_id][patient_not_na==1].mean(axis=0).squeeze(),
            dict_arrays["metabolites"][patient_id][patient_not_na==1].mean(axis=0).squeeze(),
            dict_arrays["static_patient_features"][patient_id][patient_not_na==1].mean(axis=0).squeeze(),
            np.exp(dict_arrays["y_baseline"][patient_id][patient_not_na==1]).mean(axis=0)[..., -1],
        ])
        patient_shap = mean_shap[slice_start:slice_end]
        
        # Average shap values over multiple meals
        mean_patient_shap = patient_shap.mean(axis=0)

        slice_start += sum_notna

        # make shap explanation object
        explanation = shap.Explanation(
            values=mean_patient_shap,
            base_values=mean_base_value,
            data=mean_patient_data,
            feature_names=all_features_names
        )

        fig = plt.figure()
        shap.plots.waterfall(explanation, show=False, max_display=25)
        fig.set_size_inches(20, 15)  # change after because waterfall resize the fig
        # plt.show()
        fig.savefig(f"{path_patient_plots}/features_shapley_time_{time_point+1}.pdf", format="pdf")
        plt.close()

print("\n ---------------- END ------------------")
