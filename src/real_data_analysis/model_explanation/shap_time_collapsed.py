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
with open(os.path.join(PATH_RESULTS, "all_scalers"), "rb") as fp:   # Pickling scalers
    all_scalers = pickle.load(fp)

with open(os.path.join(PATH_RESULTS, "all_shap_values"), "rb") as fp:
    all_shap_values = pickle.load(fp)
print("Shap length: ", len(all_shap_values))

# pre-process the input data with all folds scalers at once
dict_shap = {key: array for key, array in dict_arrays.items() if key in config_dict["data_arrays"].keys()}
features_combined, features_label_per_folds = prepare_data_for_shap(
    dict_shap,
    all_scalers,
    config_dict
)


# Collapse shapley values to one single time
len(all_shap_values[0])
n_patients = all_shap_values[0][0].shape[0]

feature_shap_values = [all_shap_values[t][0] for t in range(n_timepoints)]

concat_shap_values = np.concatenate(feature_shap_values, axis=-1)
concat_shap_values.shape

mean_abs_shap_values = np.mean(np.abs(concat_shap_values), axis=-1)
mean_abs_shap_values.shape
mean_shap_values = np.mean(concat_shap_values, axis=-1)
sum_shap_values = np.sum(concat_shap_values, axis=-1)

# genes shap
explanation = shap.Explanation(
    values=mean_abs_shap_values,
    data=features_combined[0],
    feature_names=genes_names
)
shap.plots.bar(explanation[0], max_display=25)

explanation = shap.Explanation(
    values=sum_shap_values,
    data=features_combined[0],
    feature_names=genes_names
)
shap.plots.bar(explanation, max_display=25)
shap.plots.beeswarm(explanation, max_display=25)
shap.plots.bar(explanation[0], max_display=25)

# long
long_shap_values = np.concatenate(feature_shap_values, axis=0)
long_shap_values.shape
long_features_combined = np.concatenate([features_combined[0] for t in range(n_timepoints)], axis=0)

explanation = shap.Explanation(
    values=long_shap_values[..., -1],
    data=long_features_combined,
    feature_names=genes_names
)
shap.plots.bar(explanation, max_display=25)
shap.plots.beeswarm(explanation, max_display=25)
shap.plots.bar(explanation[0], max_display=25)

time_groups = np.concatenate(
    [np.repeat(f"time_{tt}", n_patients) for tt in range(n_timepoints)],
    axis=0
)
shap.plots.bar(explanation.cohorts(time_groups))

explanation = shap.Explanation(
    values=long_shap_values[..., -1],
    data=long_features_combined,
    feature_names=genes_names
)
shap.plots.bar(explanation[[0, 1]].cohorts(["1", "2"]))


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
patient_cols.append("Baseline")

for time_point in range(n_timepoints):
    patient_data_shap = all_shap_values[time_point][2][...,-1]
    patient_data_shap = np.concatenate([patient_data_shap, all_shap_values[time_point][3][...,-1]], axis=-1)

    features = np.concatenate([features_combined[2], features_combined[3]], axis=-1)

    fig = plt.figure()
    shap.summary_plot(
        patient_data_shap,
        features=features,
        feature_names=patient_cols,
        show=False
    )
    fig.savefig(f"{PATH_PLOTS}/patient_features_shap_time_{time_point}.pdf", format="pdf")

print("\n ------------ END --------------")
