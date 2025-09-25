# Analyse output from TSD
import os
import pickle

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

from src.vae_attention.full_model import DeltaTimeAttentionVAE
from real_data_analysis.convert_to_array import convert_to_static_multidim_array, convert_to_longitudinal_multidim_array
from real_data_analysis.features_preprocessing import preprocess, preprocess_transform
from src.utils import data_loading_wrappers
from src.utils import plots


# Create directory for results
PATH_MODELS = "./real_data_analysis/res"

# --------------------------------------------------------
# -------------- Script Parameters --------------
# --------------------------------------------------------

# column names
PATIENT_ID = "ID"
PATIENT_MEAL_ID = "ID_Meal"
COL_MEAL = "Meal"
COL_VISIT = "Visit"
COL_TIME = "Time"
COL_OUTCOME = "TG"

COL_SEX = "Sex"
COL_AGE = "Age"
COL_BMI = "BMI"
PATIENT_COLS = [COL_SEX, COL_AGE, COL_BMI]

FEATURES_KEYS = ["X", "X_static", "y_baseline"]

# script parameters
SAVE_MODELS = True
N_FOLDS = 10
DEVICE = torch.device("cpu")
# model params
LATENT_DIM = 10
TRANSFORMER_INPUT_DIM = 256
TRANSFORMER_DIM_FEEDFORWARD = TRANSFORMER_INPUT_DIM * 4


# Load attention weights
with open(f"{PATH_MODELS}/attn_weights", "rb") as fp:
    attn_weights = pickle.load(fp)

attn_weights.shape
plots.plot_attention_weights(attn_weights, observation=0, layer_name="Attention")


# Load Correlations of genes prediction
with open(f"{PATH_MODELS}/correlations", "rb") as fp:
    correlations = pickle.load(fp)
correlations.shape


with open(f"{PATH_MODELS}/array_folds_perturbations", "rb") as fp:
    array_folds_perturbations = pickle.load(fp)
array_folds_perturbations.shape

mean_perturbations = array_folds_perturbations.mean(axis=0)
top_sensitive_features = np.argsort(mean_perturbations, axis=1)
latent_dim = 0
print(f"Features most sensitive to latent dim {latent_dim}: \n {top_sensitive_features[latent_dim, :]}")
