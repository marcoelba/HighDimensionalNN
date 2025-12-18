# Data analysis
import pickle
import os
import copy
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from src.utils.convert_to_array import convert_to_static_multidim_array, convert_to_longitudinal_multidim_array
from src.utils.features_preprocessing import preprocess_train, preprocess_transform
from src.utils.config_reader import read_config
from src.utils.get_arrays import load_and_process_data
from src.utils import training_wrapper
from src.utils import data_loading_wrappers

# Script specific modules
# Must be in the same directory where model_fitting.py is run
from full_model import Model


# read input arguments from console
parser = argparse.ArgumentParser(description='Run program with custom config and modules')
parser.add_argument('-c', '--config', required=True, help='Path to config.ini file')
args = parser.parse_args()

# Load config file
config_path = Path(args.config)
if not config_path.exists():
    print(f"Error: Config file not found: {config_path}")
    sys.exit(1)
config_dict = read_config(config_path)

PATH_RESULTS = config_dict["script_parameters"]["results_folder"]
DEVICE = torch.device(config_dict["training_parameters"]["device"])
os.makedirs(config_dict["script_parameters"]["results_folder"], exist_ok = True)

# --------------------------------------------------------
# -------------------- Load data -------------------------
# --------------------------------------------------------
dict_arrays = load_and_process_data(config_dict, data_dir=config_dict["script_parameters"]["data_folder"])
n_individuals = dict_arrays["genes"].shape[0]
p_gene = dict_arrays["genes"].shape[2]
p_metab = dict_arrays["metabolites"].shape[2]
p_static = dict_arrays["static_patient_features"].shape[2]
n_timepoints = dict_arrays["y_target"].shape[2]

model_paramerers = dict(
    p_gene=p_gene,
    p_metab=p_metab,
    p_static=p_static,
    n_timepoints=n_timepoints
)
with open(f"{PATH_RESULTS}/model_paramerers", "wb") as fp:   # Pickling predictions
    pickle.dump(model_paramerers, fp)

print("\n preprocessing dict: ", config_dict["preprocess"])

# ------------- k-fold Cross-Validation -------------
all_train_losses = []
all_val_losses = []
all_predictions = []
all_true = []
all_models = []
all_best_epochs = []
all_scalers = []

train_indices = np.random.permutation(np.arange(0, n_individuals))
# Split into k folds
folds = np.array_split(train_indices, config_dict["training_parameters"]["n_folds"])

for fold in range(config_dict["training_parameters"]["n_folds"]):
    print(f"Running k-fold validation on fold {fold+1} of {config_dict['training_parameters']['n_folds']}")

    # mask current fold for use in validation
    train_mask = np.ones(n_individuals, dtype=int)
    train_mask[folds[fold]] = False

    # Split
    dict_train = {name: arr[train_mask == 1] for name, arr in dict_arrays.items()}
    dict_val = {name: arr[train_mask == 0] for name, arr in dict_arrays.items()}

    # train and apply feature preprocessing
    dict_train_preproc, dict_val_preproc, scalers = preprocess_train(dict_train, dict_val, config_dict)
    all_scalers.append(scalers)
    
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
        batch_size=config_dict["training_parameters"]["batch_size"],
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
    model = Model(
        input_dim_genes=p_gene,
        input_dim_metab=p_metab,
        input_patient_features_dim=p_static,
        n_timepoints=n_timepoints,
        model_config=config_dict["model_params"]
    ).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print("\n Total number of model parameters: ", sum([param.numel() for param in model.parameters()]))
    # Training Loop
    trainer = training_wrapper.Training(train_dataloader, val_dataloader, noisy_gradient=False)
    trainer.training_loop(model, optimizer, config_dict["training_parameters"]["num_epochs"], gradient_noise_std=0.0)

    # use the model at the best validation iteration
    model.load_state_dict(trainer.best_model.state_dict())

    # save model?
    if config_dict["training_parameters"]["save_models"]:
        PATH = f"{PATH_RESULTS}/model_{fold}"
        torch.save(model.state_dict(), PATH)

    all_models.append(model)
    all_best_epochs.append(trainer.best_iteration)

    # Validate
    model.eval()
    with torch.no_grad():
        pred = model(val_dataloader.dataset.arrays)
        all_predictions.append(pred[-1].numpy())
        all_true.append(val_dataloader.dataset.arrays[-1].numpy())
    
    # store
    all_train_losses.append(np.min(trainer.losses["train"]))
    all_val_losses.append(np.min(trainer.losses["val"]))

    # save also plot of train/validation losses for inspection
    fig = plt.figure()
    plt.plot(trainer.losses["train"], label="Train")
    plt.plot(trainer.losses["val"], label="Val")
    plt.legend()
    fig.savefig(f"{PATH_RESULTS}/train_val_loss_fold_{fold}.pdf", format="pdf")
    plt.close()

    # only prediction loss
    fig = plt.figure()
    plt.plot(trainer.losses["train_pred"], label="Train")
    plt.plot(trainer.losses["val_pred"], label="Val")
    plt.legend()
    fig.savefig(f"{PATH_RESULTS}/train_val_prediction_loss_fold_{fold}.pdf", format="pdf")
    plt.close()

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
with open(f"{PATH_RESULTS}/predictions", "wb") as fp:   # Pickling predictions
    pickle.dump(predictions, fp)
with open(f"{PATH_RESULTS}/ground_truth", "wb") as fp:   # Pickling ground truth
    pickle.dump(ground_truth, fp)
with open(f"{PATH_RESULTS}/best_epochs", "wb") as fp:   # Pickling best epochs
    pickle.dump(all_best_epochs, fp)
with open(f"{PATH_RESULTS}/all_train_losses", "wb") as fp:   # Pickling train losses
    pickle.dump(all_train_losses, fp)
with open(f"{PATH_RESULTS}/all_val_losses", "wb") as fp:   # Pickling val losses
    pickle.dump(all_val_losses, fp)
with open(f"{PATH_RESULTS}/all_scalers", "wb") as fp:   # Pickling scalers
    pickle.dump(all_scalers, fp)

print("\n ---------------------------------------")
print(" ---------- Script finished ----------")
print(" ---------------------------------------")
