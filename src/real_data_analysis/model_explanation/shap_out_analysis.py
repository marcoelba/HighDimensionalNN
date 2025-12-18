# SHAP output analysis
import os

import shap
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.utils.config_reader import get_config
from src.utils.get_arrays import CustomData
from src.utils.prepare_data_for_shap import prepare_data_for_shap


# get config from console input arguments
config_dict = get_config()
PATH_PLOTS = config_dict["script_parameters"]["shap_plots_folder"]
os.makedirs(PATH_PLOTS, exist_ok = True)

# load model pipeline data
pipeline_data = EnsemblePipelineData(config_dict=config_dict)
all_scalers = pipeline_data.load_scalers()
all_shap_values = pipeline_data.load_shap_values()
model_paramerers = pipeline_data.load_model_paramerers()

# Load data
data = CustomData(config_dict, data_dir=config_dict["script_parameters"]["data_folder"])
dict_arrays = data.load_and_process_data(data_dir=config_dict["script_parameters"]["data_folder"])

# pre-process the input data with all folds scalers at once
features_combined, features_label_per_folds = prepare_data_for_shap(
    dict_arrays,
    all_scalers,
    config_dict
)

print("--------------------- Making Plots ------------------------")

# genes shap
for time_point in range(model_paramerers["n_timepoints"]):
    genes_shap = all_shap_values[time_point][0][..., -1]
    fig = plt.figure()
    shap.summary_plot(
        genes_shap,
        features=features_combined[0],
        feature_names=data.features_names["genes_names"],
        show=False
    )
    plt.title(f"Genes shapley values - Time {time_point}", fontsize=10)
    fig.savefig(f"{PATH_PLOTS}/genes_shap_time_{time_point}.pdf", format="pdf")

# metab shap
for time_point in range(model_paramerers["n_timepoints"]):
    metab_shap = all_shap_values[time_point][1][..., -1]
    fig = plt.figure()
    shap.summary_plot(
        metab_shap,
        features=features_combined[1],
        feature_names=data.features_names["metab_names"],
        show=False
    )
    plt.title(f"Metabolites shapley values - Time {time_point}", fontsize=10)
    fig.savefig(f"{PATH_PLOTS}/metab_shap_time_{time_point}.pdf", format="pdf")

# patient features shap
patient_cols = config_dict["data_arrays"]["static_patient_features"]
patient_cols.append("Baseline")

for time_point in range(model_paramerers["n_timepoints"]):
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
    plt.title(f"Patient features shapley values - Time {time_point}", fontsize=10)
    fig.savefig(f"{PATH_PLOTS}/patient_features_shap_time_{time_point}.pdf", format="pdf")

print("\n ------------ END --------------")
