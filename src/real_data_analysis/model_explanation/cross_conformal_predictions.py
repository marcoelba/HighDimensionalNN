# Cross-Conformal Predictions
import os

import numpy as np
import pandas as pd

from src.utils.config_reader import get_config
from src.utils.data_handling.data_loader import CustomData
from src.utils.ensemble_pipeline import EnsemblePipeline
from src.utils.features_preprocessing import Preprocessing
from src.utils.files_io_helpers import load_pickle, save_pickle

# Script specific modules
# Must be in the same directory where model_fitting.py is run
from full_model import Model


# get config from console input arguments
config_dict = get_config()
PATH_RESULTS = config_dict["script_parameters"]["results_folder"]

# Load data
data = CustomData(config_dict, data_dir=config_dict["script_parameters"]["data_folder"])
dict_arrays = data.load_and_process_data(data_dir=config_dict["script_parameters"]["data_folder"])

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

# Cross-Conformal predictions
cc_tau_per_fold = []
for fold in range(config_dict["training_parameters"]["n_folds"]):
    # Compute residuals on validation fold
    # TODO: change this after rerunning the full model
    # features_preprocessing = model_pipeline.all_scalers[fold]
    # cal_preds = np.exp(features_preprocessing.scalers["y_target"][0].inverse_transform(predictions_val_folds[fold]))
    # y_true = np.exp(features_preprocessing.scalers["y_target"][0].inverse_transform(ground_truth_val_folds[fold]))
    cal_preds = predictions_val_folds[fold]
    y_true = ground_truth_val_folds[fold]
    residuals = np.abs(y_true - cal_preds)
    # Compute quantile with finite-sample correction
    m = len(residuals)
    q_idx = int(np.ceil((1 - config_dict["training_parameters"]["alpha_conf_pred"]) * (m + 1))) - 1  # 0-indexed
    q_idx = min(q_idx, m - 1)
    tau = np.sort(residuals, axis=0)[q_idx]
    cc_tau_per_fold.append(tau)
# Save pickle with the estimated CC std
save_pickle(
    cc_tau_per_fold,
    os.path.join(PATH_RESULTS, config_dict["saving_file_names"]["pickle_cc_tau"])
)
