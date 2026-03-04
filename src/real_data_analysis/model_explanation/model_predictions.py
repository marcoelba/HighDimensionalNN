# Patient specific predictions and shap explanations
import os

import numpy as np
import matplotlib.pyplot as plt

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
PATH_PLOTS = config_dict["script_parameters"]["patient_specific_plots_folder"]
os.makedirs(PATH_PLOTS, exist_ok = True)

# Load data
data = CustomData(config_dict, data_dir=config_dict["script_parameters"]["data_folder"])
dict_arrays = data.load_and_process_data(data_dir=config_dict["script_parameters"]["data_folder"])
meal_idx_mapping = data.meal_idx_mapping

# Load Cross-Conformal Tau
cc_tau_per_fold = load_pickle(
    os.path.join(PATH_RESULTS, config_dict["saving_file_names"]["pickle_cc_tau"])
)

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
# take exp again
ground_truth = np.exp(ground_truth)
baseline = np.exp(baseline)

# prediction on all folds
predictions_folds, predictions_folds_original = model_pipeline.predict(dict_arrays)
predictions_folds_original = np.exp(predictions_folds_original)
average_prediction = np.array(predictions_folds_original).mean(axis=0)

# Conformal intervals
L_vals = []
U_vals = []
for fold, pred_fold in enumerate(predictions_folds):
    l_bound = pred_fold - cc_tau_per_fold[fold]
    u_bound = pred_fold + cc_tau_per_fold[fold]
    # transform back
    features_preprocessing = model_pipeline.all_scalers[fold]
    l_bound = np.exp(model_pipeline.target_inv_transform(features_preprocessing, l_bound, reshape=True))
    u_bound = np.exp(model_pipeline.target_inv_transform(features_preprocessing, u_bound, reshape=True))
    # Store bounds
    L_vals.append(l_bound)
    U_vals.append(u_bound)
# Compute final intervals (median across folds)
L_vals = np.stack(L_vals, axis=0)
U_vals = np.stack(U_vals, axis=0)
lower_bounds = np.nanmedian(L_vals, axis=0)
upper_bounds = np.nanmedian(U_vals, axis=0)

# Add the baseline
gt = np.concatenate([baseline, ground_truth], axis=-1)
pred = np.concatenate([baseline, average_prediction], axis=-1)

legend_meal = np.array(list(meal_idx_mapping.keys()))
colors_seq = np.array(["tab:blue", "tab:orange", "tab:green", "tab:red"])
n_times = pred.shape[-1]
n_meals = pred.shape[1]
x_ticks = range(0, n_times)
x_labels = [f"{t*2}H" if t > 0 else "Baseline" for t in range(0, n_times)]
cc_x_axis = np.arange(1, n_times, step=1)
# cc_x_axis_per_meal = cc_x_axis - np.linspace(start=0, stop=0.02, num=n_meals)[..., None]
cc_x_axis_per_meal = cc_x_axis

# -------------------- Individual Prediction plots ---------------------
figure_name = "original_scale_prediction"
title = "TG prediction"

for patient_id in range(gt.shape[0]):
    # make folder for patient specific plots
    path_patient_plots = f"{PATH_PLOTS}/patient_{patient_id}"
    os.makedirs(path_patient_plots, exist_ok = True)

    patient_gt = gt[patient_id]
    patient_pred = pred[patient_id]
    # CC bounds
    patient_lb = lower_bounds[patient_id]
    patient_ub = upper_bounds[patient_id]

    # Calculate global y-axis limits for this patient
    # Get all valid (non-NaN) values across all meals
    all_values = []
    for meal in range(n_meals):
        if np.isnan(patient_gt[meal]).sum() == 0:  # If meal has valid data
            all_values.extend(patient_gt[meal])
            all_values.extend(patient_pred[meal])
            if config_dict['training_parameters']['use_cc_predictions']:
                all_values.extend(patient_lb[meal])  # Add lower bound
                all_values.extend(patient_ub[meal])  # Add upper bound
    
    # Calculate global min and max with a small padding (optional)
    if all_values:  # Make sure we have values
        y_min = np.nanmin(all_values)
        y_max = np.nanmax(all_values)
        # Add 5% padding to make plots look better
        y_range = y_max - y_min
        y_min = y_min - 0.05 * y_range
        y_max = y_max + 0.05 * y_range
    else:
        y_min, y_max = 0, 1  # Fallback if no data

    # plot of true and predicted trajectories
    n_good_meals = (~np.isnan(patient_gt[:, 0])).sum()
    meal_counter = 0
    if n_good_meals == 1:
        height_fig = 4
    else:
        height_fig = 3 * n_good_meals
    fig, axs = plt.subplots(nrows=n_good_meals, figsize=(6.5, height_fig), ncols=1, sharex=True)
    for meal in range(n_meals):
        if np.isnan(patient_gt[meal]).sum() == 0:
            if n_good_meals > 1:
                ax = axs[meal_counter]
            else:
                ax = axs
            ax.plot(patient_gt[meal], color=colors_seq[meal], label=legend_meal[meal])
            ax.plot(patient_pred[meal], color=colors_seq[meal], linestyle="dashed")
            if config_dict['training_parameters']['use_cc_predictions']:
                ax.vlines(
                    x=cc_x_axis,
                    ymin=patient_lb[meal],
                    ymax=patient_ub[meal],
                    color=colors_seq[meal],
                    linestyle='dotted'
                )
            ax.legend(loc="best")
            ax.set_ylim(y_min, y_max)
            meal_counter += 1
    plt.subplots_adjust(hspace=0.0)
    plt.xticks(x_ticks, x_labels)
    plt.xlabel("Time")
    plt.suptitle(title)
    fig.savefig(f"{path_patient_plots}/{figure_name}.pdf", format="pdf")
    plt.close()


# -------------------- Average Prediction plots ---------------------
gt_per_meal = np.nanmean(gt, axis=0)
pred_per_meal = np.nanmean(pred, axis=0)
# CC bounds
meals_lb = np.nanmean(lower_bounds, axis=0)
meals_ub = np.nanmean(upper_bounds, axis=0)

figure_name = "original_scale_prediction_per_meal"
title = "Aggregated TG predictions per meal"

fig = plt.figure()
for meal in range(n_meals):
    plt.plot(gt_per_meal[meal], color=colors_seq[meal], label=legend_meal[meal])
    plt.plot(pred_per_meal[meal], color=colors_seq[meal], linestyle="dashed")
plt.legend(loc="best")
plt.xticks(x_ticks, x_labels)
plt.xlabel("Time")
plt.title(title)
fig.savefig(f"{PATH_RESULTS}/{figure_name}.pdf", format="pdf")
plt.close()


# -------------------- Overall Average Prediction plot ---------------------
gt_mean = np.nanmean(gt, axis=(0, 1))
pred_mean = np.nanmean(pred, axis=(0, 1))
# CC bounds
mean_lb = np.nanmean(lower_bounds, axis=(0, 1))
mean_ub = np.nanmean(upper_bounds, axis=(0, 1))

figure_name = "original_scale_prediction_average"
title = "Aggregated TG predictions overall"

fig = plt.figure()
plt.plot(gt_mean, color="blue", label="Ground truth")
plt.plot(pred_mean, color="blue", linestyle="dashed", label="Prediction")
plt.xticks(x_ticks, x_labels)
plt.xlabel("Time")
plt.legend()
plt.title(title)
fig.savefig(f"{PATH_RESULTS}/{figure_name}.pdf", format="pdf")
plt.close()

print("\n ---------------- END ------------------")
