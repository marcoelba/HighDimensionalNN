# SHAP explanations
import pickle

import shap
import torch
import numpy as np

from src.utils.config_reader import get_config
from src.utils.get_arrays import CustomData
from src.utils.prepare_data_for_shap import prepare_data_for_shap
from src.utils.ensemble_pipeline import EnsemblePipelineData, EnsembleModelSingleTime

# Script specific modules
# Must be in the same directory where model_fitting.py is run
from full_model import Model


# get config from console input arguments
config_dict = get_config()
pipeline_data = EnsemblePipelineData(config_dict=config_dict)
all_scalers = pipeline_data.load_scalers()
model_paramerers = pipeline_data.load_model_paramerers()

# Load data
data = CustomData(config_dict, data_dir=config_dict["script_parameters"]["data_folder"])
dict_arrays = data.load_and_process_data(data_dir=config_dict["script_parameters"]["data_folder"])

# make ensemble model
ensemble_model = EnsembleModelSingleTime(
    full_model=Model,
    config_dict=config_dict,
    model_paramerers=model_paramerers,
    all_scalers=all_scalers
)

# make shap arrays
features_combined, features_label_per_folds = prepare_data_for_shap(
    dict_arrays,
    all_scalers,
    config_dict,
    verbose=False
)
print("Shape background_data for SHAP: ", features_combined[0].shape)

# Run SHAP explanation
print("---------------- Running SHAP ---------------")
all_shap_values = []
for time_point in range(model_paramerers["n_timepoints"]):
    ensemble_model.time_to_explain = time_point
    explainer = shap.GradientExplainer(ensemble_model, features_combined)
    shap_values = explainer.shap_values(features_combined)
    all_shap_values.append(shap_values)

# save shap values to pickle
with open(f"{config_dict["script_parameters"]["results_folder"]}/all_shap_values", "wb") as fp:
    pickle.dump(all_shap_values, fp)
