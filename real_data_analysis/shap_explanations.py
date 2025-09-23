# SHAP explanations
import pickle
import os

import shap
import torch
import pandas as pd
import numpy as np

from src.vae_attention.full_model import DeltaTimeAttentionVAE
from real_data_analysis.convert_to_array import convert_to_static_multidim_array, convert_to_longitudinal_multidim_array
from real_data_analysis.features_preprocessing import preprocess, preprocess_transform
from src.utils import data_loading_wrappers


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

# Load torch models
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


# Define a Torch ensemble model that takes in input a list of models
class EnsembleModel(torch.nn.Module):
    def __init__(self, model_list):
        super(EnsembleModel, self).__init__()
        self.models = torch.nn.ModuleList(model_list)
        
    def forward(self, x):
        """
        Args:
            x: torch tensor array with ALL features concatenated
        """
        # Transform input back to a list of tensors
        tensors_list = []
        start_s = 0
        end_s = 0
        for ii, key in enumerate(FEATURES_KEYS):
            n_feat = len(features[key].values())
            end_s += n_feat
            tensors_list.append(x[:, start_s:end_s])
            start_s += n_feat
        # x is expected to be already properly scaled
        outputs = [model(tensors_list)[1] for model in self.models]
        return torch.stack(outputs).mean(dim=0)


# SHAP needs as input a single numpy array with the features
# pre-process the input data with all folds scalers at once
def prepare_data_for_shap(dict_shap):
    tensor_input_per_fold = []

    for fold in range(N_FOLDS):
        # apply feature preprocessing
        dict_arrays_preproc = preprocess_transform(
            dict_shap, all_scalers[fold], features_to_preprocess
        )
        if dict_arrays_preproc["y_baseline"].shape[-1] == 1:
            dict_arrays_preproc["y_baseline"] = dict_arrays_preproc["y_baseline"][..., 0]
        
        tensor_data = [
            torch.FloatTensor(array).to(DEVICE) for key, array in dict_arrays_preproc.items() if key in FEATURES_KEYS
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
    
    combined_tensors_per_fold = []
    for fold in range(N_FOLDS):
        combined_tensors_per_fold.append(torch.cat(tensor_input_per_fold[fold], dim=1))    
    combined_tensors = torch.cat(combined_tensors_per_fold, dim=0)

    return combined_tensors


# -------------------------------------------------------------------------
# ---------------------- Run SHAP explanation -----------------------------
# -------------------------------------------------------------------------
print("---------------- Running SHAP ---------------")
ensemble_model = EnsembleModel(all_models)
dict_shap = {key: array for key, array in dict_arrays.items() if key in FEATURES_KEYS}

background_data = prepare_data_for_shap(dict_shap)
print("Shape background_data for SHAP: ", background_data.shape)
explainer = shap.GradientExplainer(ensemble_model, background_data)

shap_values = explainer.shap_values(background_data)

# save shap values ot pickle
with open(f"{PATH_MODELS}/shap_values", "wb") as fp:
    pickle.dump(shap_values, fp)
