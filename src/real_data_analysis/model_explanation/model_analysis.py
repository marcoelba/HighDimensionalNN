# analysis of loss components
import os

import numpy as np
import pandas as pd

from src.utils.config_reader import get_config
from src.utils.data_handling.data_loader import CustomData
from src.utils.ensemble_pipeline import EnsemblePipeline
from src.utils.features_preprocessing import Preprocessing
from src.utils.files_io_helpers import load_pickle

# Script specific modules
# Must be in the same directory where model_fitting.py is run
from full_model import Model


# get config from console input arguments
config_dict = get_config()
PATH_RESULTS = config_dict["script_parameters"]["results_folder"]

# Load data
data = CustomData(config_dict, data_dir=config_dict["script_parameters"]["data_folder"])
dict_arrays = data.load_and_process_data(data_dir=config_dict["script_parameters"]["data_folder"])
meal_idx_mapping = data.meal_idx_mapping

model_pipeline = EnsemblePipeline(
    Model,
    Preprocessing,
    config_dict
)
# load trained models
model_pipeline.load_trained_models()
# check that the scalers have been uploaded
assert len(model_pipeline.all_scalers) > 0

# Load predictions on validation sets
predictions_val_folds = load_pickle(
    os.path.join(PATH_RESULTS, config_dict["saving_file_names"]["pickle_predictions_val"])
)
ground_truth_val_folds = load_pickle(
    os.path.join(PATH_RESULTS, config_dict["saving_file_names"]["pickle_ground_truth_val"])
)
# pred_val = np.exp(np.concatenate(predictions_val_folds, axis=0))
# gt_val = np.exp(np.concatenate(ground_truth_val_folds, axis=0))
# print(f"RMSE Validation Predictions: {np.sqrt(np.nanmean((gt_val - pred_val)**2))}")
# print(f"RMSE Validation Predictions per Time: {np.sqrt(np.nanmean((gt_val - pred_val)**2, axis=0))}")

# TODO: remove this when fitting a new model next time
predictions_val = []
ground_truth_val = []
for fold in range(config_dict["training_parameters"]["n_folds"]):
    features_preprocessing = model_pipeline.all_scalers[fold]
    y_pred = predictions_val_folds[fold]
    y_true = ground_truth_val_folds[fold]
    # inverse transform pred and gt
    predictions_val.append(features_preprocessing.scalers["y_target"][0].inverse_transform(y_pred))
    ground_truth_val.append(features_preprocessing.scalers["y_target"][0].inverse_transform(y_true))
pred_val = np.exp(np.concatenate(predictions_val, axis=0))
gt_val = np.exp(np.concatenate(ground_truth_val, axis=0))
print(f"RMSE Validation Predictions: {np.sqrt(np.nanmean((gt_val - pred_val)**2))}")
print(f"RMSE Validation Predictions per Time: {np.sqrt(np.nanmean((gt_val - pred_val)**2, axis=0))}")
# END

# target
ground_truth = dict_arrays["y_target"][..., -1]
# take exp again
ground_truth = np.exp(ground_truth)

# prediction on all folds
_, predictions_folds_original = model_pipeline.predict(dict_arrays)
average_prediction = np.array(np.exp(predictions_folds_original)).mean(axis=0)

# RMSE
rmse = np.sqrt(np.nanmean((ground_truth - average_prediction)**2))
print(f"\nRMSE overall:\n{rmse}")

rmse_per_meal = np.sqrt(np.nanmean((ground_truth - average_prediction)**2, axis=(0, 2)))
df_rmse_per_meal = pd.DataFrame(rmse_per_meal, index=meal_idx_mapping, columns=["RMSE"])
print(f"\nRMSE per meal:\n{df_rmse_per_meal}")

rmse_per_meal_time = np.sqrt(np.nanmean((ground_truth - average_prediction)**2, axis=0))
df_rmse_per_meal_time = pd.DataFrame(
    rmse_per_meal_time,
    index=meal_idx_mapping,
    columns=[f"t{t+1}" for t in range(data.n_timepoints)]
)
print(f"\nRMSE per meal, per time:\n{df_rmse_per_meal_time}")
