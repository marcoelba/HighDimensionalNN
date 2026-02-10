# SHAP output analysis
import os

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
PATH_PLOTS = config_dict["script_parameters"]["shap_plots_folder"]
os.makedirs(PATH_PLOTS, exist_ok = True)

# Load data
data = CustomData(config_dict, data_dir=config_dict["script_parameters"]["data_folder"])
dict_arrays = data.load_and_process_data(data_dir=config_dict["script_parameters"]["data_folder"])

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
time_labels = [f"{t*2}H" for t in range(1, n_timepoints+1)]

# Load shapley values
shapley_values_per_feature = load_pickle(
        os.path.join(PATH_RESULTS, config_dict["saving_file_names"]["pickle_shapley_values_arrays"])
)
# array structure: (n_folds x batch x n_meals x n_feat x n_timepoints)

# genes
for time_point in range(n_timepoints):
    shap_summary_plot_feature(
        [shapley_values_per_feature[0]],
        [dict_arrays["genes"]],
        [data.features_names["genes_names"]],
        time_point=time_point,
        title="Genes",
        time_label=time_labels[time_point],
        path_plots=PATH_PLOTS
    )

# metabolites
for time_point in range(n_timepoints):
    shap_summary_plot_feature(
        [shapley_values_per_feature[1]],
        [dict_arrays["metabolites"]],
        [data.features_names["metab_names"]],
        time_point=time_point,
        title="Metabolites",
        time_label=time_labels[time_point],
        path_plots=PATH_PLOTS
    )

# Control variables
for time_point in range(n_timepoints):
    shap_summary_plot_feature(
        [shapley_values_per_feature[2], shapley_values_per_feature[3]],
        [dict_arrays["static_patient_features"], dict_arrays["y_baseline"]],
        [config_dict["data_arrays"]["static_patient_features"], ["Baseline"]],
        time_point=time_point,
        title="Control variables",
        time_label=time_labels[time_point],
        path_plots=PATH_PLOTS
    )

print("\n ------------ END --------------")
