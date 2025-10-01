# Analyse output from TSD
import os
import pickle

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import shap

from src.vae_attention.full_model import DeltaTimeAttentionVAE
from real_data_analysis.convert_to_array import convert_to_static_multidim_array, convert_to_longitudinal_multidim_array
from real_data_analysis.features_preprocessing import preprocess, preprocess_transform
from src.utils import data_loading_wrappers
from src.utils import plots
from real_data_analysis.config_reader import read_config


# Create directory for results
PATH_MODELS = "./real_data_analysis/res"

# --------------------------------------------------------
# -------------- Script Parameters --------------
# --------------------------------------------------------

config_dict = read_config("./real_data_analysis/config.ini")
DEVICE = torch.device(config_dict["training_parameters"]["device"])

genes_names = pd.read_csv(os.path.join(PATH_MODELS, "real_genes_names.csv"), header=0, sep=";")
genes_names = genes_names["column_names"].to_numpy()

# Load attention weights
with open(f"{PATH_MODELS}/attn_weights", "rb") as fp:
    attn_weights = pickle.load(fp)

attn_weights.shape
plots.plot_attention_weights(attn_weights, observation=0, layer_name="Attention")


# Load Correlations of genes prediction
with open(f"{PATH_MODELS}/correlations", "rb") as fp:
    correlations_genes = pickle.load(fp)
correlations_genes.shape
sorted_corr = np.argsort(correlations_genes)
sorted_corr[0:10]
genes_names[sorted_corr[0:10]]  # genes less well reconstructed
genes_names[sorted_corr[-10:]]  # genes better reconstructed


with open(f"{PATH_MODELS}/array_folds_perturbations", "rb") as fp:
    array_folds_perturbations = pickle.load(fp)
array_folds_perturbations.shape


mean_perturbations = array_folds_perturbations.mean(axis=0)
mean_perturbations.shape

latent_dim = 1
sorted_mean_perturbations = np.argsort(mean_perturbations[latent_dim, :])[::-1]
genes_names[sorted_mean_perturbations[0:10]]  # top 10

plt.scatter(y=mean_perturbations[latent_dim, sorted_mean_perturbations], x=range(len(genes_names)))
plt.show()

np.corrcoef(mean_perturbations, rowvar=True)

# SHAP output
with open(f"{PATH_MODELS}/shap_values", "rb") as fp:
    shap_values = pickle.load(fp)
len(shap_values)
genes_shap = shap_values[0]
genes_shap.shape
patient_data_shap = shap_values[1]
baseline_shap = shap_values[2]

n_timepoints = genes_shap.shape[-1]

# genes shap
time_point = 0
fig = plt.figure()
shap.summary_plot(genes_shap[:, :, time_point], feature_names=genes_names, show = False)
fig.show()
# fig.savefig(f"genes_shap_time_{time_point}.pdf", format="pdf")

sorted_genes = np.argsort(np.abs(genes_shap[:, :, time_point]).sum(axis=0))[::-1]
top_genes = sorted_genes[0:5]

shap.summary_plot(genes_shap[:, top_genes[0], :], feature_names=range(n_timepoints), show = True)
shap.summary_plot(genes_shap[:, top_genes[1], :], feature_names=range(n_timepoints), show = True)
shap.summary_plot(genes_shap[:, top_genes[2], :], feature_names=range(n_timepoints), show = True)

fig = plt.figure()
plt.violinplot(genes_shap[:, top_genes, time_point])
plt.xlabel("Latent Space Dimensions")
# set style for the axes
labels = range(latent_dim)
fig.axes[0].set_xticks(np.arange(1, len(labels) + 1), labels=labels)
fig.show()


# Patient features
patient_cols = config_dict["column_names"]["patient_cols"]
patient_data_shap.shape

time_point = 0
shap.summary_plot(patient_data_shap[:, :, time_point], feature_names=patient_cols, show = True)

shap.summary_plot(patient_data_shap[:, 0, :], feature_names=range(n_timepoints), show = True)
shap.summary_plot(patient_data_shap[:, 1, :], feature_names=range(n_timepoints), show = True)
shap.summary_plot(patient_data_shap[:, 2, :], feature_names=range(n_timepoints), show = True)

# Baseline TG
baseline_shap.shape
time_point = 0

shap.summary_plot(baseline_shap[:, :, time_point], feature_names=["Baseline TG"], show = True)

shap.summary_plot(baseline_shap[:, 0, :], feature_names=range(n_timepoints), show = True)
