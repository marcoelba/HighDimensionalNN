# SHAP explanations
import os

import shap
import numpy as np

from src.utils.config_reader import get_config
from src.utils.data_handling.data_loader import CustomData
from src.utils.ensemble_pipeline import EnsemblePipeline
from src.utils.features_preprocessing import Preprocessing
from src.utils.shap.prepare_data_for_shap import prepare_data_for_shap
from src.utils.shap.shap_ensemble_pipeline import ShapEnsembleModelSingleTime, ShapEnsembleModel
from src.utils.files_io_helpers import save_pickle

# Script specific modules
# Must be in the same directory where model_fitting.py is run
from full_model import Model


# get config from console input arguments
config_dict = get_config()
path_results = config_dict["script_parameters"]["results_folder"]

# Load data
data = CustomData(config_dict, data_dir=config_dict["script_parameters"]["data_folder"])
dict_arrays = data.load_and_process_data(data_dir=config_dict["script_parameters"]["data_folder"])

# model
model_pipeline = EnsemblePipeline(
    Model,
    Preprocessing,
    config_dict
)
# load trained models
model_pipeline.load_trained_models()
# check that the scalers have been uploaded
assert len(model_pipeline.all_scalers) > 0
all_scalers = model_pipeline.all_scalers

# make ensemble model
shap_model = ShapEnsembleModel(
    model_pipeline=model_pipeline,
    config_dict=config_dict
)
preprocessed_features_flat_per_fold, preprocessed_features_per_fold, predictions_per_fold = shap_model.preprocessing(dict_arrays)
# save predictions for the base data
save_pickle(
    predictions_per_fold,
    os.path.join(path_results, f"{config_dict["saving_file_names"]["pickle_predictions_base_shapley"]}")
)

# Run SHAP explanation
print("---------------- Running SHAP ---------------")
n_timepoints = model_pipeline.model_dimension_definition["n_timepoints"]
n_folds = config_dict["training_parameters"]["n_folds"]
all_shap_values = []
for fold in range(n_folds):
    shap_model.fold_to_explain = fold
    fold_shap_values = []
    for time_point in range(n_timepoints):
        shap_model.time_to_explain = time_point
        explainer = shap.GradientExplainer(shap_model, preprocessed_features_flat_per_fold[fold])
        shap_values = explainer.shap_values(preprocessed_features_flat_per_fold[fold])
        # Reshape
        shap_values_reshaped = []
        for jj, shap_feature_flat in enumerate(shap_values):
            shap_model.tensor_not_na_indexes[jj]
            shap_feature = np.zeros(shap_model.features_flat_shape[jj]) * np.nan
            shap_feature[shap_model.tensor_not_na_indexes[jj]] = shap_feature_flat[..., -1]
            shap_values_reshaped.append(shap_feature.reshape(shap_model.features_shape[jj]))
        fold_shap_values.append(shap_values_reshaped)
    # stack time points as additional last dimension for each feature array
    shap_time_array = []
    for jj in range(len(shap_values)):
        shap_time_array.append(np.stack([fold_shap_values[time_point][jj] for time_point in range(n_timepoints)], axis=-1))
    all_shap_values.append(shap_time_array)
# make one array with stacked folds, final shape: (n_folds x batch x n_meals x n_feat x n_timepoints)
shapley_values_per_feature = []
for jj in range(len(shap_values)):
    shapley_values_per_feature.append(np.stack([all_shap_values[fold][jj] for fold in range(n_folds)], axis=0))

# save shap values to pickle
save_pickle(
    all_shap_values,
    os.path.join(path_results, f"{config_dict["saving_file_names"]["pickle_shapley_values"]}")
)
save_pickle(
    shapley_values_per_feature,
    os.path.join(path_results, f"{config_dict["saving_file_names"]["pickle_shapley_values_arrays"]}")
)

print("---------------------- END --------------------------")
