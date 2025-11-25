# SHAP output analysis
import pickle
import os
import copy
import re
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

PATH_PLOTS = config_dict["script_parameters"]["shap_plots_folder"]
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
with open(os.path.join(PATH_RESULTS, "genes_latent_shap_values"), "rb") as fp:
    genes_latent_shap = pickle.load(fp)
print("genese shapley shape: ", genes_latent_shap.shape)

with open(os.path.join(PATH_RESULTS, "metab_latent_shap_values"), "rb") as fp:
    metab_latent_shap = pickle.load(fp)
print("metabs shapley shape: ", metab_latent_shap.shape)

# -------------- Try clustering of shapley values -----------------

# genes
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler

genes_latent_shap_sum = np.abs(genes_latent_shap).mean(axis=0)
genes_latent_shap_df = pd.DataFrame(genes_latent_shap_sum, index=genes_names)
genes_latent_shap_df.to_csv(os.path.join(PATH_RESULTS, "genes_latent_shap.csv"), sep=";")

hdb_genes = HDBSCAN(min_cluster_size=2)
hdb_genes.fit(genes_latent_shap_sum)

genes_clusters = np.concatenate([
    hdb_genes.labels_[..., None],
    hdb_genes.probabilities_[..., None]
    ], axis=1
)
genes_clusters_df = pd.DataFrame(genes_clusters, columns=["Cluster", "Prob"], index=genes_names)
genes_clusters_df.to_csv(os.path.join(PATH_RESULTS, "genes_clusters.csv"), sep=";")

print("Number of clusters: ", sum(np.unique(hdb_genes.labels_) >= 0))
print("Number of clusters including outliers: ", len(np.unique(hdb_genes.labels_)))

# metabolites
metab_latent_shap_sum = np.abs(metab_latent_shap).mean(axis=0)
metab_latent_shap_df = pd.DataFrame(metab_latent_shap_sum, index=metab_names)
metab_latent_shap_df.to_csv(os.path.join(PATH_RESULTS, "metab_latent_shap.csv"), sep=";")

hdb_metab = HDBSCAN(min_cluster_size=2)
hdb_metab.fit(metab_latent_shap_sum)

metab_clusters = np.concatenate([
    hdb_metab.labels_[..., None],
    hdb_metab.probabilities_[..., None]
    ], axis=1
)
metab_clusters_df = pd.DataFrame(metab_clusters, columns=["Cluster", "Prob"], index=metab_names)
metab_clusters_df.to_csv(os.path.join(PATH_RESULTS, "metab_clusters.csv"), sep=";")

print("Number of clusters: ", sum(np.unique(hdb_metab.labels_) >= 0))
print("Number of clusters including outliers: ", len(np.unique(hdb_metab.labels_)))


print("\n ------------ END --------------")
