# Data analysis
import numpy as np
import pandas as pd
import torch

import os
os.chdir("./")

from real_data_analysis.convert_to_array import convert_to_multidim_array_efficient

from src.utils import training_wrapper
from src.utils import data_loading_wrappers
from src.utils.model_output_details import count_parameters
from src.utils import plots

from src.vae_attention.full_model import DeltaTimeAttentionVAE


# --------------------------------------------------------
# -------------- Script Parameters --------------
# --------------------------------------------------------
n_folds = 10


# Load the raw data
df = pd.read_csv('data.csv')

covs_cols = names(df)[7:end]

# convert to arrays
convert_to_multidim_array_efficient(
    df,
    patient_ID_col="ID",
    meal_col="Meal",
    time_index_col="Time",
    covs_cols=cov_cols,
    outcome_col="TG"
)

print("y and X extracted!")

n_individuals, n_timepoints, n_measurements = y.shape
p = X.shape[1]

print("Dimensions:")
print("n_individuals: ", n_individuals)
print("n_timepoints: ", n_timepoints)
print("n_measurements: ", n_measurements)
print("p: ", p)

# y0 (y at baseline) is actually an additional feature, because it is measured before any intervention
y.shape
y_baseline = y[:, :, 0:1]
y_baseline.shape
# the actual target is then y from t=1
y_target = y[:, :, 1:]
y_target.shape

n_timepoints = n_timepoints -1

# get tensors
X_tensor = torch.FloatTensor(X).to(torch.device("cpu"))
X_static_tensor = torch.FloatTensor(X_static).to(torch.device("cpu"))
y_target_tensor = torch.FloatTensor(y_target).to(torch.device("cpu"))
y_baseline_tensor = torch.FloatTensor(y_baseline).to(torch.device("cpu"))

# make tensor data loaders
reshape = True
drop_missing = True

train_dataloader = data_loading_wrappers.make_data_loader(
    *tensor_data_train, batch_size=batch_size, feature_dimensions=-1, reshape=reshape, drop_missing=drop_missing
)
test_dataloader = data_loading_wrappers.make_data_loader(
    *tensor_data_test, batch_size=batch_size, feature_dimensions=-1, reshape=reshape, drop_missing=drop_missing
)
val_dataloader = data_loading_wrappers.make_data_loader(
    *tensor_data_val, batch_size=batch_size, feature_dimensions=-1, reshape=reshape, drop_missing=drop_missing
)


# ------------- k-fold Cross-Validation -------------

all_train_losses = []
all_val_losses = []
all_predictions = []
all_true = []
all_models = []

train_indices = np.random.permutation(np.arange(0, n_train))
# Split into k folds
folds = np.array_split(train_indices, n_folds)
num_epochs = 500

for fold in range(n_folds):
    print(f"Running k-fold validation on fold {fold+1} of {n_folds}")

    train_mask = torch.ones(n_train, dtype=torch.bool)
    train_mask[folds[fold]] = False

    # make train and validation data loader for the LOO cross-validation
    tensor_train_loo = [
        tensor_data_train[0][train_mask],
        tensor_data_train[1][train_mask]
    ]
    tensor_val_loo = [
        tensor_data_train[0][~train_mask],
        tensor_data_train[1][~train_mask],
    ]
    train_dataloader = data_loading_wrappers.make_data_loader(*tensor_train_loo, batch_size=batch_size)
    val_dataloader = data_loading_wrappers.make_data_loader(*tensor_val_loo, batch_size=1)

    # 4. Training Setup
    model = NNModel(p1, ldim=50, dropout_prob=0.2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training Loop
    trainer = training_wrapper.Training(train_dataloader, val_dataloader, noisy_gradient=False)
    trainer.training_loop(model, optimizer, num_epochs, gradient_noise_std=0.0)

    # use the model at the best validation iteration
    model.load_state_dict(trainer.best_model.state_dict())
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
