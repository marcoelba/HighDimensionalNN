# Latent space SHAP explanations
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

PATH_PLOTS = config_dict["script_parameters"]["latent_shap_plots_folder"]
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

genes_names = pd.read_csv(os.path.join(PATH_DATA, "genes_names.csv"), header=0, sep=";")
genes_names = genes_names["column_names"].to_numpy()
metab_names = pd.read_csv(os.path.join(PATH_DATA, "metab_names.csv"), header=0, sep=";")
metab_names = metab_names["column_names"].to_numpy()

# Load pickle files
with open(f"{PATH_RESULTS}/all_scalers", "rb") as fp:   # Pickling scalers
    all_scalers = pickle.load(fp)

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


# Define a Torch ensemble model that takes in input a list of models
class EnsembleModelGenes(torch.nn.Module):
    def __init__(self, model_list, torch_scalers_outcome=None):
        super(EnsembleModelGenes, self).__init__()
        self.models = torch.nn.ModuleList(model_list)
        self.torch_scalers_outcome = torch_scalers_outcome

    def forward(self, x):
        all_outputs = []
        for fold, model in enumerate(self.models):
            model.eval()
            output = model.ffn_genes(x)
            all_outputs.append(output)
        return torch.stack(all_outputs).mean(dim=0)


class EnsembleModelMetabs(torch.nn.Module):
    def __init__(self, model_list, torch_scalers_outcome=None):
        super(EnsembleModelMetabs, self).__init__()
        self.models = torch.nn.ModuleList(model_list)
        self.torch_scalers_outcome = torch_scalers_outcome

    def forward(self, x):
        all_outputs = []
        for fold, model in enumerate(self.models):
            model.eval()
            output = model.ffn_metab(x)
            all_outputs.append(output)
        return torch.stack(all_outputs).mean(dim=0)

# -------------------------------------------------------------------------
# ---------------------- Run SHAP explanation -----------------------------
# -------------------------------------------------------------------------
print("---------------- Running SHAP ---------------")
dict_shap = {key: array for key, array in dict_arrays.items() if key in config_dict["data_arrays"].keys()}
features_combined, features_label_per_folds = prepare_data_for_shap(
    dict_shap,
    all_scalers,
    config_dict,
    verbose=False
)
print("Shape background_data for SHAP: ", features_combined[0].shape)

# Genes
ensemble_model = EnsembleModelGenes(all_models)
explainer = shap.GradientExplainer(ensemble_model, features_combined[0])
genes_shap_values = explainer.shap_values(features_combined[0])

# save shap values to pickle
with open(f"{PATH_RESULTS}/genes_latent_shap_values", "wb") as fp:
    pickle.dump(genes_shap_values, fp)

# Metabolites
ensemble_model = EnsembleModelMetabs(all_models)
explainer = shap.GradientExplainer(ensemble_model, features_combined[1])
metab_shap_values = explainer.shap_values(features_combined[1])

# save shap values to pickle
with open(f"{PATH_RESULTS}/metab_latent_shap_values", "wb") as fp:
    pickle.dump(metab_shap_values, fp)

# ----------------------------------------------------------------------
# ---------------------------- Plots -----------------------------------
# ----------------------------------------------------------------------

# genes shap
for latent in range(config_dict["model_params"]["vae_genomics_latent_dim"]):
    fig = plt.figure()
    shap.summary_plot(
        genes_shap_values[..., latent],
        features=features_combined[0],
        feature_names=genes_names,
        show=False
    )
    fig.savefig(f"{PATH_PLOTS}/genes_latent_{latent}.pdf", format="pdf")
    plt.close()

# metab shap
for latent in range(config_dict["model_params"]["vae_metabolomics_latent_dim"]):
    fig = plt.figure()
    shap.summary_plot(
        metab_shap_values[..., latent],
        features=features_combined[1],
        feature_names=metab_names,
        show=False
    )
    fig.savefig(f"{PATH_PLOTS}/metab_latent_{latent}.pdf", format="pdf")
    plt.close()
