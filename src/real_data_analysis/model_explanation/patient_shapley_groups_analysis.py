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
n_patients = dict_arrays[array_names[0]].shape[0]

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
pathway_genes_dict = load_pickle(
        os.path.join(PATH_RESULTS, "pathway_genes_dict")
)
genes_pathway_dict = load_pickle(
        os.path.join(PATH_RESULTS, "genes_pathway_dict")
)
pathway_metabs_dict = load_pickle(
        os.path.join(PATH_RESULTS, "pathway_metabs_dict")
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

# feature names
genes_names = data.features_names[array_names[0]]
metab_names = data.features_names[array_names[1]]
control_names = control_vars_to_plot
baseline_names = np.array([data.features_names[array_names[3]]])

# Prepare shapley values aggregate over folds, meals and time (0, 2, -1)
genes_shap = np.nanmean(shapley_values_per_feature[0], axis=(0, 2, -1))
metab_shap = np.nanmean(shapley_values_per_feature[1], axis=(0, 2, -1))
control_shap = np.nanmean(shapley_values_per_feature[2][:, :, :, indices_vars_to_plot, :], axis=(0, 2, -1))
baseline_shap = np.nanmean(shapley_values_per_feature[3], axis=(0, 2, -1))

# do not include reference genes
ref_genes = pathway_genes_dict["Reference Genes"]
pos_no_ref_genes = np.intersect1d(
    genes_names, np.setdiff1d(genes_names, ref_genes), return_indices=True
)[1]

# --------------------------------------------------------------------
# Total over omics variables

all_features_names = np.concatenate([
    np.array(["Genes"]),
    np.array(["Metabolites"]),
    control_names,
    baseline_names
])

# Concatenate the features, by taking the mean over meals (1)
all_features = np.concatenate([
    np.repeat(np.nan, n_patients)[..., None],
    np.repeat(np.nan, n_patients)[..., None],
    np.nanmean(dict_arrays[array_names[2]][..., indices_vars_to_plot], axis=1),
    np.nanmean(np.exp(dict_arrays[array_names[3]]), axis=1)[..., -1]
    ], axis=-1
)

# Use sum over omics
# shape each array (n_folds x batch x n_meals x n_feat x n_timepoints)
# take mean over folds, meals and time

# sum over omics WITH ABS
all_shapley_values = np.concatenate([
    np.sum(np.abs(genes_shap), axis=-1)[..., None],
    np.sum(np.abs(metab_shap), axis=-1)[..., None],
    np.abs(control_shap),
    np.abs(baseline_shap)
    ], axis=-1
)

# Plot shap waterfall values
for patient_id in range(all_features.shape[0]):
    # make folder for patient specific plots
    path_patient_plots = f"{PATH_PLOTS}/patient_{patient_id}"
    os.makedirs(path_patient_plots, exist_ok = True)

    patient_shapley_values = all_shapley_values[patient_id]
    patient_features = all_features[patient_id]

    # make shap explanation object
    explanation = shap.Explanation(
        values=patient_shapley_values,
        data=patient_features,
        feature_names=all_features_names
    )

    fig = plt.figure()
    shap.plots.bar(explanation, show=False, max_display=25)
    fig.set_size_inches(18, 8)  # change after because waterfall resize the fig
    # plt.show()
    plt.title(f"Patient {patient_id} - Total Omics Absolute Shapley values", loc='left', fontsize=20)
    fig.savefig(f"{path_patient_plots}/total_omics_abs_shapley.pdf", format="pdf")
    plt.close()


# sum over omics withOUT ABS
all_shapley_values = np.concatenate([
    np.sum(genes_shap, axis=-1)[..., None],
    np.sum(metab_shap, axis=-1)[..., None],
    control_shap,
    baseline_shap
    ], axis=-1
)

# Plot shap waterfall values
for patient_id in range(all_features.shape[0]):
    # make folder for patient specific plots
    path_patient_plots = f"{PATH_PLOTS}/patient_{patient_id}"
    os.makedirs(path_patient_plots, exist_ok = True)

    patient_shapley_values = all_shapley_values[patient_id]
    patient_features = all_features[patient_id]

    # make shap explanation object
    explanation = shap.Explanation(
        values=patient_shapley_values,
        data=patient_features,
        feature_names=all_features_names
    )

    fig = plt.figure()
    shap.plots.bar(explanation, show=False, max_display=25)
    fig.set_size_inches(18, 8)  # change after because waterfall resize the fig
    # plt.show()
    plt.title(f"Patient {patient_id} - Total Omics Shapley values", loc='left', fontsize=20)
    fig.savefig(f"{path_patient_plots}/total_omics_shapley.pdf", format="pdf")
    plt.close()


# --------------------------------------------------------------------
# Using omics pathways
genes_shap = np.nanmean(shapley_values_per_feature[0], axis=(0, 2, -1))
metab_shap = np.nanmean(shapley_values_per_feature[1], axis=(0, 2, -1))
control_shap = np.nanmean(shapley_values_per_feature[2][:, :, :, indices_vars_to_plot, :], axis=(0, 2, -1))
baseline_shap = np.nanmean(shapley_values_per_feature[3], axis=(0, 2, -1))

# Aggregate metabolites - NO ABS
metab_shap_groups = []
for groupm, metabs_in_group in pathway_metabs_dict.items():
    where = np.intersect1d(metab_names, metabs_in_group, return_indices=True)[1]
    agg_shap = np.nansum(metab_shap[:, where], axis=-1)
    metab_shap_groups.append(agg_shap)
metab_shap_groups = np.stack(metab_shap_groups, axis=-1)

# Aggregate genes in pathways
genes_shap_groups = []
non_empty_pathways = []
for pathway, genes in pathway_genes_dict.items():
    where = np.intersect1d(genes_names, genes, return_indices=True)[1]
    int_genes = np.intersect1d(genes_names, genes)
    n_pathways_per_gene = np.array([len(genes_pathway_dict[gene]) for gene in genes if gene in int_genes])
    agg_shap = genes_shap[:, where] / n_pathways_per_gene
    sum_agg_shap = np.nansum(agg_shap, axis=-1)
    if len(int_genes) > 0:
        non_empty_pathways.append(pathway)
        genes_shap_groups.append(sum_agg_shap)
genes_shap_groups = np.stack(genes_shap_groups, axis=-1)

# feature names
all_features_names = np.concatenate([
    np.array(non_empty_pathways),
    np.array(list(pathway_metabs_dict.keys())),
    control_names,
    baseline_names
])

# Concatenate the features, by taking the mean over meals (1)
all_features = np.concatenate([
    np.ones([n_patients, len(non_empty_pathways)]) * np.nan,
    np.ones([n_patients, len(pathway_metabs_dict.keys())]) * np.nan,
    np.nanmean(dict_arrays[array_names[2]][..., indices_vars_to_plot], axis=1),
    np.nanmean(np.exp(dict_arrays[array_names[3]]), axis=1)[..., -1]
    ], axis=-1
)

# sum over omics WITH-OUT ABS
all_shapley_values = np.concatenate([
    genes_shap_groups,
    metab_shap_groups,
    control_shap,
    baseline_shap
    ], axis=-1
)

# Plot shap waterfall values
for patient_id in range(all_features.shape[0]):
    # make folder for patient specific plots
    path_patient_plots = f"{PATH_PLOTS}/patient_{patient_id}"
    os.makedirs(path_patient_plots, exist_ok = True)

    patient_shapley_values = all_shapley_values[patient_id]
    patient_features = all_features[patient_id]

    # make shap explanation object
    explanation = shap.Explanation(
        values=patient_shapley_values,
        data=patient_features,
        feature_names=all_features_names
    )

    fig = plt.figure()
    shap.plots.bar(explanation, show=False, max_display=25)
    fig.set_size_inches(20, 15)  # change after because waterfall resize the fig
    # plt.show()
    plt.title(f"Patient {patient_id} - Pathways Omics Shapley values", loc='left', fontsize=20)
    fig.savefig(f"{path_patient_plots}/pathways_omics_abs_shapley.pdf", format="pdf")
    plt.close()

print("\n ---------------- END ------------------")
