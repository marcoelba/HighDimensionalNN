# SHAP output analysis
import pickle
import os
import copy
import re

import shap
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from real_data_analysis.utils.convert_to_array import convert_to_static_multidim_array, convert_to_longitudinal_multidim_array
from real_data_analysis.utils.features_preprocessing import preprocess_train, preprocess_transform
from real_data_analysis.utils.prepare_data_for_shap import prepare_data_for_shap

from real_data_analysis.model_genes_metabolomics.get_arrays import load_and_process_data
from real_data_analysis.model_genes_metabolomics.config_reader import read_config
from real_data_analysis.model_genes_metabolomics.full_model import DeltaTimeAttentionVAE

from src.utils import data_loading_wrappers


# Create directory for results
PATH_MODELS = "./real_data_analysis/results/res_train_v3"
PATH_PLOTS = "plots"
os.makedirs(PATH_PLOTS, exist_ok = True)

# --------------------------------------------------------
# -------------- Script Parameters --------------
# --------------------------------------------------------

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

genes_names = pd.read_csv("./real_data_analysis/data/genes_names.csv", header=0, sep=";")
genes_names = genes_names["column_names"].to_numpy()
metab_names = pd.read_csv("./real_data_analysis/data/metab_names.csv", header=0, sep=";")
metab_names = metab_names["column_names"].to_numpy()

# Load pickle files
with open(f"{PATH_MODELS}/all_scalers", "rb") as fp:   # Pickling scalers
    all_scalers = pickle.load(fp)

with open(f"{PATH_MODELS}/all_shap_values", "rb") as fp:
    all_shap_values = pickle.load(fp)
print("Shap length: ", len(all_shap_values))
all_shap_values[0][0].shape

# pre-process the input data with all folds scalers at once
dict_shap = {key: array for key, array in dict_arrays.items() if key in config_dict["data_arrays"].keys()}
features_combined, features_label_per_folds = prepare_data_for_shap(
    dict_shap,
    all_scalers,
    config_dict
)

# genes shap
for time_point in range(n_timepoints):
    genes_shap = all_shap_values[time_point][0][..., -1]
    fig = plt.figure()
    shap.summary_plot(
        genes_shap,
        features=features_combined[0],
        feature_names=genes_names,
        show=False
    )
    fig.savefig(f"{PATH_PLOTS}/genes_shap_time_{time_point}.pdf", format="pdf")

# metab shap
for time_point in range(n_timepoints):
    metab_shap = all_shap_values[time_point][1][..., -1]
    fig = plt.figure()
    shap.summary_plot(
        metab_shap,
        features=features_combined[1],
        feature_names=metab_names,
        show=False
    )
    fig.savefig(f"{PATH_PLOTS}/metab_shap_time_{time_point}.pdf", format="pdf")

# patient features shap
patient_cols = config_dict["data_arrays"]["static_patient_features"]

for time_point in range(n_timepoints):
    patient_data_shap = shap_values[time_point][2][...,-1]
    fig = plt.figure()
    shap.summary_plot(
        patient_data_shap,
        features=features_combined[2],
        feature_names=patient_cols,
        show=False
    )
    fig.savefig(f"{PATH_PLOTS}/patient_features_shap_time_{time_point}.pdf", format="pdf")

print("\n ------------ END --------------")
