# Data analysis
import numpy as np
import pandas as pd
import torch

import os
os.chdir("./")

from real_data_analysis.convert_to_array import convert_to_static_multidim_array, convert_to_longitudinal_multidim_array

from src.utils import training_wrapper
from src.utils import data_loading_wrappers
from src.utils.model_output_details import count_parameters
from src.utils import plots

from src.vae_attention.full_model import DeltaTimeAttentionVAE


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

# script parameters
SAVE_MODELS = True
PATH_MODELS = "./"
N_FOLDS = 10
BATCH_SIZE = 50
NUM_EPOCHS = 500

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

n_individuals, n_measurements, n_timepoints = y.shape
p = X.shape[1]
p_static = X_static.shape[1]

print("Dimensions:")
print("n_individuals: ", n_individuals)
print("n_timepoints: ", n_timepoints)
print("n_measurements: ", n_measurements)
print("p: ", p)
print("p_static: ", p_static)

# y0 (y at baseline) is actually an additional feature, because it is measured before any intervention
print(y.shape)
y_baseline = y[:, :, 0:1]
print(y_baseline.shape)
# the actual target is then y from t=1
y_target = y[:, :, 1:]
print(y_target.shape)

n_timepoints = n_timepoints - 1
print("n_timepoints withOUT baseline: ", n_timepoints)

# get tensors
X_tensor = torch.FloatTensor(X).to(torch.device("cpu"))
X_static_tensor = torch.FloatTensor(X_static).to(torch.device("cpu"))
y_target_tensor = torch.FloatTensor(y_target).to(torch.device("cpu"))
y_baseline_tensor = torch.FloatTensor(y_baseline).to(torch.device("cpu"))
print("\n Tensors created")

# make list of tensors
tensor_data_train = [
    X_tensor,
    X_static_tensor,
    y_baseline_tensor,
    y_target_tensor
]

# make tensor data loaders
reshape = True
drop_missing = True

# ------------- k-fold Cross-Validation -------------
all_train_losses = []
all_val_losses = []
all_predictions = []
all_true = []
all_models = []

train_indices = np.random.permutation(np.arange(0, n_individuals))

# Split into k folds
folds = np.array_split(train_indices, N_FOLDS)

for fold in range(N_FOLDS):
    print(f"Running k-fold validation on fold {fold+1} of {N_FOLDS}")

    train_mask = torch.ones(n_individuals, dtype=torch.bool)
    train_mask[folds[fold]] = False

    # make train and validation data loader for the k-fold cross-validation
    tensor_train_loo = [
        tensor_data_train[0][train_mask],
        tensor_data_train[1][train_mask]
    ]
    tensor_val_loo = [
        tensor_data_train[0][~train_mask],
        tensor_data_train[1][~train_mask],
    ]
    # data loaders
    train_dataloader = data_loading_wrappers.make_data_loader(
        *tensor_train_loo,
        batch_size=BATCH_SIZE,
        feature_dimensions=-1,
        reshape=reshape,
        drop_missing=drop_missing
    )
    val_dataloader = data_loading_wrappers.make_data_loader(
        *tensor_val_loo,
        batch_size=1,
        feature_dimensions=-1,
        reshape=reshape,
        drop_missing=drop_missing
    )

    # ---------------------- Model Setup ----------------------
    device = torch.device("cpu")

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
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training Loop
    trainer = training_wrapper.Training(train_dataloader, val_dataloader, noisy_gradient=False)
    trainer.training_loop(model, optimizer, NUM_EPOCHS, gradient_noise_std=0.0)

    # use the model at the best validation iteration
    model.load_state_dict(trainer.best_model.state_dict())

    # save model?
    if save_models:
        PATH = f"{PATH_MODELS}\model_{fold}"
        torch.save(model.state_dict(), PATH)

    all_models.append(model)
    
    # Validate
    model.eval()
    with torch.no_grad():
        pred = model(tensor_val_loo)
        all_predictions.append(pred.numpy())
        all_true.append(tensor_val_loo[1].numpy())
    
    # store
    all_train_losses.append(np.min(trainer.losses["train"]))
    all_val_losses.append(np.min(trainer.losses["val"]))


predictions = np.concatenate(all_predictions, axis=0)
predictions.shape
ground_truth = np.concatenate(all_true, axis=0)
ground_truth.shape
