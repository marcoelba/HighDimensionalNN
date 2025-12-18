# Patient specific predictions and shap explanations
import os

import shap
import torch
import pandas as pd
import numpy as np

from src.utils.config_reader import get_config
from src.utils.data_handling.data_loader import CustomData
from src.utils.shap.prepare_data_for_shap import prepare_data_for_shap
from src.utils.ensemble_pipeline import EnsemblePipeline
from src.utils.plotting.target_predictions import plot_predictions

# Script specific modules
# Must be in the same directory where model_fitting.py is run
from full_model import Model


# get config from console input arguments
config_dict = get_config()

N_FOLDS = config_dict["training_parameters"]["n_folds"]
PATH_PLOTS = config_dict["script_parameters"]["patient_specific_plots_folder"]
os.makedirs(PATH_PLOTS, exist_ok = True)

# Load data
data = CustomData(config_dict, data_dir=config_dict["script_parameters"]["data_folder"])
dict_arrays = data.load_and_process_data(data_dir=config_dict["script_parameters"]["data_folder"])
where_all_non_missing = data.get_indeces(dict_arrays)

# make ensemble model
model_pipeline = EnsemblePipeline(
    Model,
    config_dict
)

features_combined, features_label_per_folds = prepare_data_for_shap(
    dict_arrays,
    model_pipeline.all_scalers,
    config_dict,
    verbose=False
)

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
    y_pred = model_pipeline.predict_fold(tensor_input, fold).numpy()
    all_predictions.append(
        np.concatenate([y_baseline, y_pred], axis=-1)
    )

# Produce plots and save
plot_predictions(
    np.array(all_predictions).mean(axis=0),
    np.array(all_ground_truth).mean(axis=0),
    data_class,
    where_all_non_missing,
    path_plots=PATH_PLOTS,
    title="Standardized-log TG",
    figure_name="prediction"
)

# plots in original scale
# inverse-transform
all_predictions_original = []
all_ground_truth_original = []
for fold in range(N_FOLDS):
    all_predictions_original.append(
        np.concatenate([
            model_pipeline.scalers[fold]["y_baseline"][0].inverse_transform(all_predictions[fold][:, 0:1]),
            model_pipeline.scalers[fold]["y_target"][0].inverse_transform(all_predictions[fold][:, 1:])
        ], axis=1)
    )
    all_ground_truth_original.append(
        np.concatenate([
            model_pipeline.scalers[fold]["y_baseline"][0].inverse_transform(all_ground_truth[fold][:, 0:1]),
            model_pipeline.scalers[fold]["y_target"][0].inverse_transform(all_ground_truth[fold][:, 1:])
        ], axis=1)
    )
all_predictions_original = np.exp(np.array(all_predictions_original))
all_ground_truth_original = np.exp(np.array(all_ground_truth_original))

plot_predictions(
    all_predictions_original.mean(axis=0),
    all_ground_truth_original.mean(axis=0),
    data_class,
    where_all_non_missing,
    title="Original scale TG",
    figure_name="original_scale_prediction"
)

print("\n ---------------- END ------------------")
