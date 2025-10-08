# Patient specific predictions and shap explanations
import pickle
import os
import copy

import shap
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from real_data_analysis.convert_to_array import convert_to_static_multidim_array, convert_to_longitudinal_multidim_array
from real_data_analysis.features_preprocessing import preprocess, preprocess_transform
from src.utils import data_loading_wrappers
from real_data_analysis.config_reader import read_config
from real_data_analysis.get_arrays import load_and_process_data

from real_data_analysis.model_use_vae_z.full_model import DeltaTimeAttentionVAE


# Read config
PATH_MODELS = "./res_train_v2"
PATH_PLOTS = "plots_patient_shap"
os.makedirs(PATH_PLOTS, exist_ok = True)

config_dict = read_config("./real_data_analysis/model_use_vae_z/config.ini")
DEVICE = torch.device(config_dict["training_parameters"]["device"])
N_FOLDS = config_dict["training_parameters"]["n_folds"]
FEATURES_KEYS = list(config_dict["preprocess"].keys())[:-1]

# --------------------------------------------------------
# -------------------- Load data -------------------------
# --------------------------------------------------------
dict_arrays, features_to_preprocess = load_and_process_data(config_dict, data_dir="./")
n_individuals = dict_arrays["genes"].shape[0]
p = dict_arrays["genes"].shape[2]
p_static = dict_arrays["static_patient_features"].shape[2]
n_timepoints = dict_arrays["y_target"].shape[2]

# change this
genes_names = pd.read_csv("genes_names.csv", header=0, sep=";")
genes_names = genes_names["column_names"].to_numpy()

all_features_names = np.concatenate([
    genes_names,
    np.array(config_dict["column_names"]["patient_cols"]),
    np.array(["Baseline"])
])

# Load pickle files
with open(f"{PATH_MODELS}/all_scalers", "rb") as fp:   # Pickling scalers
    all_scalers = pickle.load(fp)

with open(f"{PATH_MODELS}/all_shap_values", "rb") as fp:
    shap_values = pickle.load(fp)
print("Shap length: ", len(shap_values))

# Load torch models
all_models = []
for fold in range(N_FOLDS):
    print(f"Loading model fold {fold+1} of {N_FOLDS}")

    PATH = f"{PATH_MODELS}/model_{fold}"
    model = DeltaTimeAttentionVAE(
        input_dim=p,
        patient_features_dim=p_static,
        n_timepoints=n_timepoints,
        vae_latent_dim=config_dict["model_params"]["latent_dim"],
        vae_input_to_latent_dim=64,
        max_len_position_enc=10,
        transformer_input_dim=config_dict["model_params"]["transformer_input_dim"],
        transformer_dim_feedforward=config_dict["model_params"]["transformer_dim_feedforward"],
        nheads=config_dict["model_params"]["n_heads"],
        dropout=0.1,
        dropout_attention=0.1,
        prediction_weight=1.0
    ).to(DEVICE)
    model.load_state_dict(torch.load(PATH))
    all_models.append(model)


# Define a Torch ensemble model that takes in input a list of models
class EnsembleModel(torch.nn.Module):
    def __init__(self, model_list, time_to_explain):
        super(EnsembleModel, self).__init__()
        self.models = torch.nn.ModuleList(model_list)
        self.time_to_explain = time_to_explain

    def forward(self, *x):
        x_list = list(x)
        all_outputs = []
        for model in self.models:
            model.eval()
            output = model(x_list)[1][:, [self.time_to_explain]]
            all_outputs.append(output)
        return torch.stack(all_outputs).mean(dim=0)

# pre-process the input data with all folds scalers at once
def prepare_data_for_shap(dict_shap, subsample=False, n_background=100, patient_index=None):
    tensor_input_per_fold = []

    for fold in range(N_FOLDS):
        # apply feature preprocessing
        dict_arrays_preproc = preprocess_transform(
            copy.deepcopy(dict_shap), all_scalers[fold], features_to_preprocess
        )
        if dict_arrays_preproc["y_baseline"].shape[-1] == 1:
            dict_arrays_preproc["y_baseline"] = dict_arrays_preproc["y_baseline"][..., 0]
            dict_arrays_preproc["y_target"] = dict_arrays_preproc["y_target"][..., 0]

        if patient_index is not None:
            tensor_data = [
                torch.FloatTensor(array[patient_index]).to(DEVICE) for key, array in dict_arrays_preproc.items()
            ]
        elif (patient_index is None) and subsample:
            index = np.random.choice(next(iter(dict_shap.values())).shape[0], n_background, replace=False)
            tensor_data = [
                torch.FloatTensor(array[index]).to(DEVICE) for key, array in dict_arrays_preproc.items()
            ]
        else:
            tensor_data = [
                torch.FloatTensor(array).to(DEVICE) for key, array in dict_arrays_preproc.items()
            ]
        
        # make longitudinal
        tensor_dataset = data_loading_wrappers.CustomDataset(
            *tensor_data,
            reshape=True,
            remove_missing=True,
            feature_dimensions=-1,
            device=DEVICE
        )
        tensor_input = tensor_dataset.arrays
        # append
        tensor_input_per_fold.append(tensor_input)
    
    combined_tensor_features_per_fold = []
    for feature in range(len(FEATURES_KEYS)):
        tensor_feature = []
        for fold in range(N_FOLDS):
            tensor_feature.append(tensor_input_per_fold[fold][feature])
        combined_tensor_features_per_fold.append(torch.cat(tensor_feature, dim=0))
    
    return combined_tensor_features_per_fold, tensor_input_per_fold


# -------------------------------------------------------------------------
# ---------------------- Run SHAP explanation -----------------------------
# -------------------------------------------------------------------------
print("---------------------------- Running SHAP -------------------------")
dict_shap = {key: array for key, array in dict_arrays.items() if key in features_to_preprocess.keys()}
background_data, tensor_input_per_fold = prepare_data_for_shap(dict_shap, subsample=False)
print("Shape background_data for SHAP: ", background_data[0].shape)

# -----------------------------------------------------------
# Get model predictions from all background data
all_predictions = []
all_ground_truth = []
for fold in range(N_FOLDS):
    
    y_true = tensor_input_per_fold[fold][-1]
    y_baseline = tensor_input_per_fold[fold][-2]
    all_ground_truth.append(
        np.concatenate([y_baseline, y_true], axis=-1)
    )

    tensor_input = tensor_input_per_fold[fold][:-1]
    with torch.no_grad():
        y_pred = all_models[fold](tensor_input)[1].numpy()

    all_predictions.append(
        np.concatenate([y_baseline, y_pred], axis=-1)
    )

# -----------------------------------------------------------
# average the predictions over folds before plotting
target_predictions = np.array(all_predictions).mean(axis=0)
target_ground_truth = np.array(all_ground_truth).mean(axis=0)

slice_start = 0
slice_end = 0
colors_seq = ["black", "red", "blue", "orange"]

for patient_id in range(n_individuals):
    # make folder for patient specific plots
    path_patient_plots = f"{PATH_PLOTS}/patient_{patient_id}"
    os.makedirs(path_patient_plots, exist_ok = True)

    patient_data = dict_shap["y_target"][patient_id]
    notna = ~np.isnan(patient_data[:, 0])
    sum_notna = notna.sum()
    slice_end += sum_notna

    # plot of true and predicted trajectories
    patient_pred = target_predictions[slice_start:slice_end]
    patient_ground_truth = target_ground_truth[slice_start:slice_end]

    fig = plt.figure()
    for meal in range(sum_notna):
        plt.plot(patient_ground_truth[meal], color=colors_seq[meal])
        plt.plot(patient_pred[meal], color=colors_seq[meal], linestyle="dashed")
    plt.xticks(range(0, n_timepoints + 1))
    plt.xlabel("Time")
    plt.title("Standardized-log TG")
    fig.savefig(f"{path_patient_plots}/predicted_y.pdf", format="pdf")
    plt.close()

# --------------------------------------------------------------
# Plot shap waterfall values
for time_point in range(n_timepoints):

    # Get base values for the explainer - time specific
    ensemble_model = EnsembleModel(all_models, time_to_explain=time_point)
    ensemble_model.eval()
    with torch.no_grad():
        predictions = ensemble_model(*background_data)
        base_values = predictions.numpy()
    mean_base_values = np.array(np.array_split(base_values, N_FOLDS, axis=0)).mean(axis=0)

    genes_shap = all_shap_values[time_point][0][..., -1]  # genes shap
    patient_clin_shap = all_shap_values[time_point][1][..., -1]  # patient shap
    baseline_shap = all_shap_values[time_point][2][..., -1]  # patient shap

    # Takes averages
    mean_genes_shap = np.array(np.array_split(genes_shap, N_FOLDS, axis=0)).mean(axis=0)
    mean_patient_clin_shap = np.array(np.array_split(patient_clin_shap, N_FOLDS, axis=0)).mean(axis=0)
    mean_baseline_shap = np.array(np.array_split(baseline_shap, N_FOLDS, axis=0)).mean(axis=0)

    genes_data = np.array(np.array_split(background_data[0], N_FOLDS, axis=0)).mean(axis=0)
    patient_clin_data = np.array(np.array_split(background_data[1], N_FOLDS, axis=0)).mean(axis=0)
    baseline_data = np.array(np.array_split(background_data[2], N_FOLDS, axis=0)).mean(axis=0)

    # make one array for shap and data
    mean_shap = np.concatenate([mean_genes_shap, mean_patient_clin_shap, mean_baseline_shap], axis=-1)
    mean_data = np.concatenate([genes_data, patient_clin_data, baseline_data], axis=-1)

    # slice over time
    slice_start = 0
    slice_end = 0
    for patient_id in range(n_individuals):
        # make folder for patient specific plots
        path_patient_plots = f"{PATH_PLOTS}/patient_{patient_id}"

        target = dict_shap["y_target"][patient_id]
        notna = ~np.isnan(target[:, 0])
        sum_notna = notna.sum()
        slice_end += sum_notna

        patient_shap = mean_shap[slice_start:slice_end]
        patient_base_value = mean_base_values[slice_start:slice_end]
        patient_data = mean_data[slice_start:slice_end]
        
        # Average shap values over multiple meals
        mean_patient_shap = patient_shap.mean(axis=0)
        mean_patient_base_value = patient_base_value.mean()
        mean_patient_data = patient_data.mean(axis=0)

        slice_start += sum_notna

        # make shap explanation object
        explanation = shap.Explanation(
            values=mean_patient_shap,
            base_values=mean_patient_base_value,
            data=mean_patient_data,  # Flatten if needed
            feature_names=all_features_names
        )

        fig = plt.figure()
        shap.plots.waterfall(explanation, show=False, max_display=25)
        fig.set_size_inches(20, 15)  # change after because waterfall resize the fig
        # plt.show()
        fig.savefig(f"{path_patient_plots}/genes_shap_time_{time_point+1}.pdf", format="pdf")
        plt.close()

print("\n ---------------- END ------------------")

# Using the latent shap values
# To group genes based on their influence on the latent dimensions
with open(f"{PATH_MODELS}/vae_latent_shap_values", "rb") as fp:
    vae_latent_shap_values = pickle.load(fp)

latent_features_indices = []
for latent_dim in range(config_dict["model_params"]["latent_dim"]):
    sum_latent = np.abs(vae_latent_shap_values[latent_dim]).sum(axis=0)
    top_10 = (sum_latent > np.quantile(sum_latent, q=0.9)).sum()
    top_10_genes = np.argsort(sum_latent)[::-1][0:top_10]
    latent_features_indices.append(top_10_genes)


new_features_names = np.concatenate([
    np.array([f"latent_dim_{ll}" for ll in range(config_dict["model_params"]["latent_dim"])]),
    np.array(config_dict["column_names"]["patient_cols"]),
    np.array(["Baseline"])
])


# Plot shap waterfall values
for time_point in range(n_timepoints):

    # Get base values for the explainer - time specific
    ensemble_model = EnsembleModel(all_models, time_to_explain=time_point)
    ensemble_model.eval()
    with torch.no_grad():
        predictions = ensemble_model(*background_data)
        base_values = predictions.numpy()
    mean_base_values = np.array(np.array_split(base_values, N_FOLDS, axis=0)).mean(axis=0)

    genes_shap = all_shap_values[time_point][0][..., -1]  # genes shap
    patient_clin_shap = all_shap_values[time_point][1][..., -1]  # patient shap
    baseline_shap = all_shap_values[time_point][2][..., -1]  # patient shap

    # Takes averages
    mean_genes_shap = np.array(np.array_split(genes_shap, N_FOLDS, axis=0)).mean(axis=0)
    mean_patient_clin_shap = np.array(np.array_split(patient_clin_shap, N_FOLDS, axis=0)).mean(axis=0)
    mean_baseline_shap = np.array(np.array_split(baseline_shap, N_FOLDS, axis=0)).mean(axis=0)

    genes_data = np.array(np.array_split(background_data[0], N_FOLDS, axis=0)).mean(axis=0)
    patient_clin_data = np.array(np.array_split(background_data[1], N_FOLDS, axis=0)).mean(axis=0)
    baseline_data = np.array(np.array_split(background_data[2], N_FOLDS, axis=0)).mean(axis=0)

    # make the average for the top genes
    new_latent_gene_shap = []
    for latent_dim in range(config_dict["model_params"]["latent_dim"]):
        new_latent_gene_shap.append(mean_genes_shap[:, latent_features_indices[latent_dim]].sum(axis=1))
    new_latent_gene_shap = np.array(new_latent_gene_shap).transpose()

    new_latent_gene_data = []
    for latent_dim in range(config_dict["model_params"]["latent_dim"]):
        new_latent_gene_data.append(genes_data[:, latent_features_indices[latent_dim]].sum(axis=1))
    new_latent_gene_data = np.array(new_latent_gene_data).transpose()

    # make one array for shap and data
    mean_shap = np.concatenate([new_latent_gene_shap, mean_patient_clin_shap, mean_baseline_shap], axis=-1)
    mean_data = np.concatenate([new_latent_gene_data, patient_clin_data, baseline_data], axis=-1)

    # slice over time
    slice_start = 0
    slice_end = 0
    for patient_id in range(n_individuals):
        # make folder for patient specific plots
        path_patient_plots = f"{PATH_PLOTS}/patient_{patient_id}"

        target = dict_shap["y_target"][patient_id]
        notna = ~np.isnan(target[:, 0])
        sum_notna = notna.sum()
        slice_end += sum_notna

        patient_shap = mean_shap[slice_start:slice_end]
        patient_base_value = mean_base_values[slice_start:slice_end]
        patient_data = mean_data[slice_start:slice_end]
        
        # Average shap values over multiple meals
        mean_patient_shap = patient_shap.mean(axis=0)
        mean_patient_base_value = patient_base_value.mean()
        mean_patient_data = patient_data.mean(axis=0)

        slice_start += sum_notna

        # make shap explanation object
        explanation = shap.Explanation(
            values=mean_patient_shap,
            base_values=mean_patient_base_value,
            data=mean_patient_data,  # Flatten if needed
            feature_names=new_features_names
        )

        fig = plt.figure()
        shap.plots.waterfall(explanation, show=False, max_display=25)
        fig.set_size_inches(20, 15)  # change after because waterfall resize the fig
        # plt.show()
        fig.savefig(f"{path_patient_plots}/latent_genes_shap_time_{time_point+1}.pdf", format="pdf")
        plt.close()

print("\n ---------------- END ------------------")
