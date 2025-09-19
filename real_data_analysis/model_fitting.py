# Data analysis
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import pickle

import os
os.chdir("./")

from real_data_analysis.convert_to_array import convert_to_static_multidim_array, convert_to_longitudinal_multidim_array
from real_data_analysis.features_preprocessing import preprocess

from src.utils import training_wrapper
from src.utils import data_loading_wrappers
# from src.utils import plots

from src.vae_attention.full_model import DeltaTimeAttentionVAE


# Create directory for results
PATH_MODELS = "./res"
os.makedirs("res", exist_ok = True)

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
N_FOLDS = 10
BATCH_SIZE = 50
BATCH_SIZE_VAL = None
NUM_EPOCHS = 500
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

# ------------- k-fold Cross-Validation -------------
all_train_losses = []
all_val_losses = []
all_predictions = []
all_true = []
all_models = []
all_best_epochs = []

train_indices = np.random.permutation(np.arange(0, n_individuals))
# Split into k folds
folds = np.array_split(train_indices, N_FOLDS)

for fold in range(N_FOLDS):
    print(f"Running k-fold validation on fold {fold+1} of {N_FOLDS}")

    # mask current fold for use in validation
    train_mask = np.ones(n_individuals, dtype=int)
    train_mask[folds[fold]] = False

    # Split
    dict_train = {name: arr[train_mask == 1] for name, arr in dict_arrays.items()}
    dict_val = {name: arr[train_mask == 0] for name, arr in dict_arrays.items()}

    # train and apply feature preprocessing
    dict_train_preproc, dict_val_preproc, scalers = preprocess(dict_train, dict_val, features_to_preprocess)

    # remove last dimension for outcome with only one dimension
    if dict_train_preproc["y_target"].shape[-1] == 1:
        dict_train_preproc["y_target"] = dict_train_preproc["y_target"][..., 0]
        dict_val_preproc["y_target"] = dict_val_preproc["y_target"][..., 0]
        dict_train_preproc["y_baseline"] = dict_train_preproc["y_baseline"][..., 0]
        dict_val_preproc["y_baseline"] = dict_val_preproc["y_baseline"][..., 0]

    # get tensors
    tensor_data_train = [torch.FloatTensor(array).to(DEVICE) for key, array in dict_train_preproc.items()]
    tensor_data_val = [torch.FloatTensor(array).to(DEVICE) for key, array in dict_val_preproc.items()]

    # Validation batch size - just do one
    y_val_shape = dict_val_preproc["y_target"].shape
    BATCH_SIZE_VAL = y_val_shape[0] * y_val_shape[1]

    # Train batch size
    y_train_shape = dict_train_preproc["y_target"].shape
    print("y_train_shape: ", y_train_shape[0] * y_train_shape[1])
    # BATCH_SIZE_TRAIN = y_train_shape[0] * y_train_shape[1]

    # data loaders
    train_dataloader = data_loading_wrappers.make_data_loader(
        *tensor_data_train,
        batch_size=BATCH_SIZE,
        feature_dimensions=-1,
        reshape=True,
        drop_missing=True
    )
    val_dataloader = data_loading_wrappers.make_data_loader(
        *tensor_data_val,
        batch_size=BATCH_SIZE_VAL,
        feature_dimensions=-1,
        reshape=True,
        drop_missing=True
    )

    # ---------------------- Model Setup ----------------------
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
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training Loop
    trainer = training_wrapper.Training(train_dataloader, val_dataloader, noisy_gradient=False)
    trainer.training_loop(model, optimizer, NUM_EPOCHS, gradient_noise_std=0.0)

    # use the model at the best validation iteration
    model.load_state_dict(trainer.best_model.state_dict())

    # save model?
    if SAVE_MODELS:
        PATH = f"{PATH_MODELS}/model_{fold}"
        torch.save(model.state_dict(), PATH)

    all_models.append(model)
    all_best_epochs.append(trainer.best_iteration)

    # Validate
    model.eval()
    with torch.no_grad():
        pred = model(tensor_data_val)
        all_predictions.append(pred[1].numpy())
        all_true.append(tensor_data_val[3].numpy())
    
    # store
    all_train_losses.append(np.min(trainer.losses["train"]))
    all_val_losses.append(np.min(trainer.losses["val"]))

#
print("\n ---------------------------------------")
print(" ---------- Training finished ----------")
print(" ---------------------------------------")

print("\n")
print("Best epochs: ", all_best_epochs)

predictions = np.concatenate(all_predictions, axis=0)
predictions.shape
ground_truth = np.concatenate(all_true, axis=0)
ground_truth.shape

# save to pickle files
with open(f"{PATH_MODELS}/predictions", "wb") as fp:   # Pickling predictions
    pickle.dump(predictions, fp)
with open(f"{PATH_MODELS}/ground_truth", "wb") as fp:   # Pickling ground truth
    pickle.dump(ground_truth, fp)
with open(f"{PATH_MODELS}/best_epochs", "wb") as fp:   #Pickling best epochs
    pickle.dump(all_best_epochs, fp)
with open(f"{PATH_MODELS}/all_train_losses", "wb") as fp:   #Pickling train losses
    pickle.dump(all_train_losses, fp)
with open(f"{PATH_MODELS}/all_val_losses", "wb") as fp:   #Pickling val losses
    pickle.dump(all_val_losses, fp)

print("\n ---------------------------------------")
print(" ---------- Script finished ----------")
print(" ---------------------------------------")
