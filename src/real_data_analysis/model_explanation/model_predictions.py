# Patient specific predictions and shap explanations
import os

import numpy as np
import matplotlib.pyplot as plt

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
PATH_PLOTS = config_dict["script_parameters"]["patient_specific_plots_folder"]
os.makedirs(PATH_PLOTS, exist_ok = True)

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

# take exp again
ground_truth = np.exp(ground_truth)
baseline = np.exp(baseline)
average_prediction = np.exp(average_prediction)

# Add the baseline
gt = np.concatenate([baseline, ground_truth], axis=-1)
pred = np.concatenate([baseline, average_prediction], axis=-1)

legend_meal = np.array(list(meal_idx_mapping.keys()))
colors_seq = np.array(["tab:blue", "tab:orange", "tab:green", "tab:red"])
n_times = pred.shape[-1]
n_meals = pred.shape[1]
x_ticks = range(0, n_times)
x_labels = [f"{t*2}H" if t > 0 else "Baseline" for t in range(0, n_times)]


# -------------------- Individual Prediction plots ---------------------
figure_name = "original_scale_prediction"
title = "TG prediction"

for patient_id in range(gt.shape[0]):
    # make folder for patient specific plots
    path_patient_plots = f"{PATH_PLOTS}/patient_{patient_id}"
    os.makedirs(path_patient_plots, exist_ok = True)

    patient_gt = gt[patient_id]
    patient_pred = pred[patient_id]

    # plot of true and predicted trajectories
    fig = plt.figure()
    for meal in range(n_meals):
        if np.isnan(patient_gt[meal]).sum() == 0:
            plt.plot(patient_gt[meal], color=colors_seq[meal], label=legend_meal[meal])
            plt.plot(patient_pred[meal], color=colors_seq[meal], linestyle="dashed")
    plt.xticks(x_ticks, x_labels)
    plt.xlabel("Time")
    plt.legend()
    plt.title(title)

    fig.savefig(f"{path_patient_plots}/{figure_name}.pdf", format="pdf")
    plt.close()


# -------------------- Average Prediction plots ---------------------
gt_per_meal = np.nanmean(gt, axis=0)
pred_per_meal = np.nanmean(pred, axis=0)

figure_name = "original_scale_prediction_per_meal"
title = "Aggregated TG predictions per meal"

fig = plt.figure()
for meal in range(n_meals):
    plt.plot(gt_per_meal[meal], color=colors_seq[meal], label=legend_meal[meal])
    plt.plot(pred_per_meal[meal], color=colors_seq[meal], linestyle="dashed")
plt.xticks(x_ticks, x_labels)
plt.xlabel("Time")
plt.legend()
plt.title(title)
fig.savefig(f"{PATH_RESULTS}/{figure_name}.pdf", format="pdf")
plt.close()


# -------------------- Overall Average Prediction plot ---------------------
gt_mean = np.nanmean(gt, axis=(0, 1))
pred_mean = np.nanmean(pred, axis=(0, 1))

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
