# Patient specific predictions and shap explanations
import os

import shap
import numpy as np
import matplotlib.pyplot as plt

from src.utils.config_reader import get_config
from src.utils.data_handling.data_loader import CustomData
from src.utils.ensemble_pipeline import EnsemblePipeline
from src.utils.features_preprocessing import Preprocessing
from src.utils.files_io_helpers import load_pickle
from src.utils.plotting.shap_plots import shap_summary_plot_feature

# Script specific modules
# Must be in the same directory where model_fitting.py is run
from full_model import Model


# get config from console input arguments
config_dict = get_config()
PATH_RESULTS = config_dict["script_parameters"]["results_folder"]
PATH_PLOTS = config_dict["script_parameters"]["patient_specific_plots_folder"]
os.makedirs(PATH_PLOTS, exist_ok = True)

# Load data
data = CustomData(config_dict, data_dir=config_dict["script_parameters"]["data_folder"])
dict_arrays = data.load_and_process_data(data_dir=config_dict["script_parameters"]["data_folder"])
array_names = list(config_dict['data_arrays'].keys())

# model
model_pipeline = EnsemblePipeline(
    Model,
    Preprocessing,
    config_dict
)
# check that the scalers have been uploaded
assert len(model_pipeline.all_scalers) > 0
all_scalers = model_pipeline.all_scalers
n_timepoints = model_pipeline.model_dimension_definition["n_timepoints"]
time_labels = [f"{t*2}H" for t in range(1, n_timepoints + 1)]

# Load shapley values
shapley_values_per_feature = load_pickle(
        os.path.join(PATH_RESULTS, config_dict["saving_file_names"]["pickle_shapley_values_arrays"])
)
# array structure per feature: (n_folds x batch x n_meals x n_feat x n_timepoints)
predictions_per_fold = load_pickle(
        os.path.join(PATH_RESULTS, config_dict["saving_file_names"]["pickle_predictions_base_shapley"])
)
# take average over folds (0) and meals (2)
predictions = np.nanmean(np.stack(predictions_per_fold), axis=(0, 2))
time_predictions_mean = predictions.mean(axis=0)

plot_meals_shap = False
control_vars_to_plot = data.features_names[array_names[2]]
if not plot_meals_shap:
    control_vars_to_plot = [(i, feat) for i, feat in enumerate(control_vars_to_plot) if "Meal" not in feat]
    indices_vars_to_plot, control_vars_to_plot = zip(*control_vars_to_plot)

all_features_names = np.concatenate([
    data.features_names[array_names[0]],
    data.features_names[array_names[1]],
    np.array(control_vars_to_plot),
    np.array([data.features_names[array_names[3]]])
])

all_shapley_values = np.concatenate([
    np.nanmean(shapley_values_per_feature[0], axis=(0, 2)),
    np.nanmean(shapley_values_per_feature[1], axis=(0, 2)),
    np.nanmean(shapley_values_per_feature[2][:, :, :, indices_vars_to_plot], axis=(0, 2)),
    np.nanmean(shapley_values_per_feature[3], axis=(0, 2))
    ], axis=-2
)

# Concatenate the features, by taking the mean over meals (second dimension)
all_features = np.concatenate([
    np.nanmean(dict_arrays[array_names[0]], axis=1),
    np.nanmean(dict_arrays[array_names[1]], axis=1),
    np.nanmean(dict_arrays[array_names[2]][..., indices_vars_to_plot], axis=1),
    np.nanmean(np.exp(dict_arrays[array_names[3]]), axis=1)[..., -1]
    ], axis=-1
)

# Plot shap waterfall values
for patient_id in range(all_features.shape[0]):
    # make folder for patient specific plots
    path_patient_plots = f"{PATH_PLOTS}/patient_{patient_id}"
    os.makedirs(path_patient_plots, exist_ok = True)

    patient_shapley_values = all_shapley_values[patient_id]
    patient_features = all_features[patient_id]
    
    for time_point in range(n_timepoints):

        time_patient_shapley_values = patient_shapley_values[..., time_point]
        
        # make shap explanation object
        explanation = shap.Explanation(
            values=time_patient_shapley_values,
            base_values=time_predictions_mean[time_point],
            data=patient_features,
            feature_names=all_features_names
        )

        fig = plt.figure()
        shap.plots.bar(explanation, show=False, max_display=25)
        fig.set_size_inches(20, 15)  # change after because waterfall resize the fig
        # plt.show()
        plt.title(f"Patient {patient_id} - Shapley values - Time {time_labels[time_point]}", loc='left', fontsize=20)
        fig.savefig(f"{path_patient_plots}/features_shapley_time_{time_point+1}.pdf", format="pdf")
        plt.close()

print("\n ---------------- END ------------------")
