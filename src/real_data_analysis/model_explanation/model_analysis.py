# analysis of loss components
import os

import numpy as np
import pandas as pd

from src.utils.config_reader import get_config
from src.utils.data_handling.data_loader import CustomData
from src.utils.ensemble_pipeline import EnsemblePipeline
from src.utils.features_preprocessing import Preprocessing

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

# target
ground_truth = dict_arrays["y_target"][..., -1]
baseline = dict_arrays["y_baseline"][..., -1]

# prediction on all folds
predictions_folds, predictions_folds_original = model_pipeline.predict(dict_arrays)
average_prediction = np.array(predictions_folds_original).mean(axis=0)
average_prediction_rescaled = np.array(predictions_folds).mean(axis=0)

# take exp again
ground_truth = np.exp(ground_truth)
baseline = np.exp(baseline)
average_prediction = np.exp(average_prediction)

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
