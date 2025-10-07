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

from real_data_analysis.convert_to_array import convert_to_static_multidim_array, convert_to_longitudinal_multidim_array
from real_data_analysis.features_preprocessing import preprocess, preprocess_transform
from src.utils import data_loading_wrappers
from real_data_analysis.config_reader import read_config
from real_data_analysis.get_arrays import load_and_process_data


# Create directory for results
PATH_MODELS = "./results"
PATH_PLOTS = "plots"
os.makedirs(PATH_PLOTS, exist_ok = True)

# --------------------------------------------------------
# -------------- Script Parameters --------------
# --------------------------------------------------------

config_dict = read_config("./model_use_vae_z/config.ini")
DEVICE = torch.device(config_dict["training_parameters"]["device"])
N_FOLDS = config_dict["training_parameters"]["n_folds"]
FEATURES_KEYS = list(config_dict["preprocess"].keys())[:-1]

def prepare_data_for_shap(dict_shap, subsample=False, n_background=100):
    tensor_input_per_fold = []

    for fold in range(N_FOLDS):
        # apply feature preprocessing
        dict_arrays_preproc = preprocess_transform(
            copy.deepcopy(dict_shap), all_scalers[fold], features_to_preprocess
        )
        if dict_arrays_preproc["y_baseline"].shape[-1] == 1:
            dict_arrays_preproc["y_baseline"] = dict_arrays_preproc["y_baseline"][..., 0]
        
        if subsample:
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

# --------------------------------------------------------
# -------------------- Load data -------------------------
# --------------------------------------------------------
dict_arrays, features_to_preprocess = load_and_process_data(config_dict, data_dir="./")
n_individuals = dict_arrays["genes"].shape[0]
p = dict_arrays["genes"].shape[2]
p_static = dict_arrays["static_patient_features"].shape[2]
n_timepoints = dict_arrays["y_target"].shape[2]

genes_names = pd.read_csv("real_genes_names.csv", header=0, sep=";")
genes_names = genes_names["column_names"].to_numpy()

# Load pickle files
with open(f"{PATH_MODELS}/all_scalers", "rb") as fp:   # Pickling scalers
    all_scalers = pickle.load(fp)

with open(f"{PATH_MODELS}/all_shap_values", "rb") as fp:
    shap_values = pickle.load(fp)
print("Shap length: ", len(shap_values))
shap_values[0][0].shape

# pre-process the input data with all folds scalers at once
dict_shap = {key: array for key, array in dict_arrays.items() if key in FEATURES_KEYS}
background_data = prepare_data_for_shap(dict_shap, subsample=False)

# genes shap
for time_point in range(n_timepoints):
    genes_shap = shap_values[time_point][0][..., -1]
    fig = plt.figure()
    shap.summary_plot(
        genes_shap,
        features=background_data[0],
        feature_names=genes_names,
        show = False
    )
    fig.savefig(f"{PATH_PLOTS}/genes_shap_time_{time_point}.pdf", format="pdf")

# Check which genes are at the top
for time_point in range(n_timepoints):
    genes_shap = shap_values[time_point][0][..., -1]
    genes_names[np.abs(genes_shap).mean(axis=0).argsort()[::-1]][0:30]


# for a single gene
time_all_genes_shap = np.concatenate([shap_values[tt][0] for tt in range(n_timepoints)], axis=2)
sorted_genes = np.argsort(np.abs(time_all_genes_shap).sum(axis=0).sum(axis=1))[::-1]
top_genes = 10

for gene in range(top_genes):
    gene_number = sorted_genes[gene]
    time_genes_shap = np.concatenate([shap_values[tt][0][:, gene_number] for tt in range(n_timepoints)], axis=1)
    x = background_data[0][:, gene_number]
    x = x.repeat(n_timepoints).reshape(time_genes_shap.shape)
    fig = plt.figure()
    shap.summary_plot(
        time_genes_shap,
        features=x,
        feature_names=[f"time_{tt}" for tt in range(n_timepoints)],
        show=False
    )
    gene_name = re.sub(r'[^a-zA-Z0-9]', '', genes_names[gene_number])
    fig.savefig(f"{PATH_PLOTS}/genes_shap_{gene_name}.pdf", format="pdf")


# patient features shap
patient_cols = config_dict["column_names"]["patient_cols"]

for time_point in range(n_timepoints):
    patient_data_shap = shap_values[time_point][1][...,-1]
    fig = plt.figure()
    shap.summary_plot(
        patient_data_shap,
        features=background_data[1],
        feature_names=patient_cols,
        show = False
    )
    fig.savefig(f"{PATH_PLOTS}/patient_features_shap_time_{time_point}.pdf", format="pdf")


print("\n ------------ END --------------")

