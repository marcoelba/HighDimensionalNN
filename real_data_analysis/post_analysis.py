# analysis of loss components
import os
import pickle

import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from src.vae_attention.full_model import DeltaTimeAttentionVAE
from real_data_analysis.convert_to_array import convert_to_static_multidim_array, convert_to_longitudinal_multidim_array
from real_data_analysis.features_preprocessing import preprocess, preprocess_transform
from src.utils import data_loading_wrappers


def loss_components(m_out, batch):
    """
    Loss function. The structure depends on the batch data.
    To be modified according to the data used.

    VAE loss + Prediction Loss (here MSE)
    """
    # Reconstruction loss (MSE)
    BCE = nn.functional.mse_loss(m_out[0], batch[0], reduction='none')
    # KL divergence
    KLD = -0.5 * torch.sum(1 + m_out[3] - m_out[2].pow(2) - m_out[3].exp())
    # label prediction loss
    PredMSE = nn.functional.mse_loss(m_out[1], batch[3], reduction='none')

    return dict(BCE=BCE, KLD=KLD, PredMSE=PredMSE)


# Create directory for results
PATH_MODELS = "./res"

# --------------------------------------------------------
# -------------- Script Parameters --------------
# --------------------------------------------------------

# File names
PATH_TO_FEATURES_DATA = "features_data_ready.csv"
PATH_TO_CLINICAL_DATA = "clinical_data_ready.csv"
PATH_TO_GENE_NAMES = "genes_names.csv"

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


# --------------------------------------------------------
# -------------- Load data --------------
# --------------------------------------------------------

# Load the patient data
df_features = pd.read_csv(PATH_TO_FEATURES_DATA, header=0, sep=";")
df_clinical_data = pd.read_csv(PATH_TO_CLINICAL_DATA, header=0, sep=";")
df_gene_names = pd.read_csv(PATH_TO_GENE_NAMES, header=0, sep=";")

# convert genomics data to array
genes_cols = df_gene_names['column_names'].tolist()

# dictionary with all feature columns
features = dict()
features["X"] = {k: v for v, k in enumerate(genes_cols)}
features["X_static"] = {k: v for v, k in enumerate(PATIENT_COLS)}
features["y_baseline"] = {"y_baseline": 0}

# add gene columns to preprocessing dictionary
features_to_preprocess = dict()
features_to_preprocess["X"] = {k: v for v, k in enumerate(genes_cols)}
features_to_preprocess["X_static"] = dict(COL_AGE=1, COL_BMI=2)

print("\n-------------------------------")
print("\nExtraction of gene data")
print("\n-------------------------------")

X = convert_to_static_multidim_array(
    df_features,
    baseline_time=0,
    patient_ID_col=PATIENT_ID,
    visit_col=COL_VISIT,
    meal_col=COL_MEAL,
    time_index_col=COL_TIME,
    cols_to_extract=genes_cols
)
print("\ngenes features extracted!")
print(X.shape)

# extract static patient features
print("\n-------------------------------")
print("\nExtraction of patient data")
print("\n-------------------------------")

X_static = convert_to_static_multidim_array(
    df_features,
    baseline_time=0,
    patient_ID_col=PATIENT_ID,
    visit_col=COL_VISIT,
    meal_col=COL_MEAL,
    time_index_col=COL_TIME,
    cols_to_extract=PATIENT_COLS
)
print("\nX_static features extracted!")
print(X_static.shape)

print("\n-------------------------------")
print("\nExtraction of outcome")
print("\n-------------------------------")

y = convert_to_longitudinal_multidim_array(
    df_clinical_data,
    patient_ID_col=PATIENT_ID,
    visit_col=COL_VISIT,
    meal_col=COL_MEAL,
    time_index_col=COL_TIME,
    cols_to_extract=[COL_OUTCOME]
)
print("\nOutcome y extracted!")
print(y.shape)

n_individuals, n_measurements, n_timepoints, _ = y.shape
p = X.shape[-1]
p_static = X_static.shape[-1]

print("Dimensions:")
print("n_individuals: ", n_individuals)
print("n_timepoints: ", n_timepoints)
print("n_measurements: ", n_measurements)
print("p: ", p)
print("p_static: ", p_static)

# y0 (y at baseline) is actually an additional feature, because it is measured before any intervention
y_baseline = y[:, :, 0:1, :]
print("Baseline: ", y_baseline.shape)
# the actual target is then y from t=1
y_target = y[:, :, 1:, :]
print("Target: ", y_target.shape)

# Add preproc info
features_to_preprocess["y_baseline"] = dict(COL_OUTCOME=0)
features_to_preprocess["y_target"] = dict(COL_OUTCOME=0)

n_timepoints = n_timepoints - 1
print("n_timepoints withOUT baseline: ", n_timepoints)

dict_arrays = dict(
    X=X,
    X_static=X_static,
    y_baseline=y_baseline,
    y_target=y_target
)

# Load pickle files
with open(f"{PATH_MODELS}/all_scalers", "rb") as fp:   # Pickling scalers
    all_scalers = pickle.load(fp)

all_models = []
for fold in range(N_FOLDS):
    print(f"Loading model fold {fold+1} of {N_FOLDS}")

    PATH = f"{PATH_MODELS}/model_{fold}"
    model = DeltaTimeAttentionVAE(
        input_dim=p,
        patient_features_dim=p_static,
        n_timepoints=n_timepoints,
        vae_latent_dim=LATENT_DIM,
        vae_input_to_latent_dim=64,
        max_len_position_enc=10,
        transformer_input_dim=TRANSFORMER_INPUT_DIM,
        transformer_dim_feedforward=TRANSFORMER_DIM_FEEDFORWARD,
        nheads=4,
        dropout=0.1,
        dropout_attention=0.1,
        prediction_weight=1.0
    ).to(DEVICE)
    model.load_state_dict(torch.load(PATH))
    all_models.append(model)


# LOSS components
mse_all_folds = []
bce_all_folds = []
attn_weights = []

for fold in range(N_FOLDS):
    # apply feature preprocessing
    dict_arrays_preproc = preprocess_transform(
        dict_arrays, all_scalers[fold], features_to_preprocess
    )
    # remove last dimension for outcome with only one dimension
    if dict_arrays_preproc["y_target"].shape[-1] == 1:
        dict_arrays_preproc["y_target"] = dict_arrays_preproc["y_target"][..., 0]
        dict_arrays_preproc["y_baseline"] = dict_arrays_preproc["y_baseline"][..., 0]

    # keep only input features and get tensors
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
    # fold-model prediction
    model = all_models[fold]
    model.eval()
    with torch.no_grad():
        pred = model(tensor_input)
        # x_hat, y_hat, mu, logvar
        loss = loss_components(
            m_out=pred,
            batch=tensor_input
        )
        mse_all_folds.append(loss["PredMSE"].numpy())
        bce_all_folds.append(loss["BCE"].numpy())

    attn_weights.append(model.get_attention_weights(tensor_input))


print("Average Prediction MSE: ", np.array(mse_all_folds).mean())
print("Average Reconstruction MSE: ", np.array(bce_all_folds).mean())

pred_mse = np.array(mse_all_folds).mean(axis=0)
pred_mse_x = np.array(bce_all_folds).mean(axis=0)
attn_weights = np.array(attn_weights).mean(axis=0)

# save attention weights
with open(f"{PATH_MODELS}/attn_weights", "wb") as fp:
    pickle.dump(attn_weights, fp)
