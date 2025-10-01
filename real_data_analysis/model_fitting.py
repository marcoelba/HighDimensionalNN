# Data analysis
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import pickle
import os
import copy
# os.chdir("./")

from real_data_analysis.convert_to_array import convert_to_static_multidim_array, convert_to_longitudinal_multidim_array
from real_data_analysis.features_preprocessing import preprocess, preprocess_transform
from real_data_analysis.config_reader import read_config
from real_data_analysis.get_arrays import load_and_process_data

from src.utils import training_wrapper
from src.utils import data_loading_wrappers
from src.vae_attention.full_model import DeltaTimeAttentionVAE


# Read config
PATH_MODELS = "./res"
os.makedirs("res", exist_ok = True)

config_dict = read_config("./real_data_analysis/config.ini")
DEVICE = torch.device(config_dict["training_parameters"]["device"])


# --------------------------------------------------------
# -------------------- Load data -------------------------
# --------------------------------------------------------
dict_arrays, features_to_preprocess = load_and_process_data(config_dict, data_dir="./real_data_analysis/res")
n_individuals = dict_arrays["genes"].shape[0]
p = dict_arrays["genes"].shape[2]
p_static = dict_arrays["static_patient_features"].shape[2]
n_timepoints = dict_arrays["y_target"].shape[2]

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
    print(f"Running k-fold validation on fold {fold+1} of {config_dict["training_parameters"]["n_folds"]}")

    # mask current fold for use in validation
    train_mask = np.ones(n_individuals, dtype=int)
    train_mask[folds[fold]] = False

    # Split
    dict_train = {name: arr[train_mask == 1] for name, arr in dict_arrays.items()}
    dict_val = {name: arr[train_mask == 0] for name, arr in dict_arrays.items()}

    # train and apply feature preprocessing
    dict_train_preproc, dict_val_preproc, scalers = preprocess(dict_train, dict_val, features_to_preprocess)
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
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training Loop
    trainer = training_wrapper.Training(train_dataloader, val_dataloader, noisy_gradient=False)
    trainer.training_loop(model, optimizer, config_dict["training_parameters"]["num_epochs"], gradient_noise_std=0.0)

    # use the model at the best validation iteration
    model.load_state_dict(trainer.best_model.state_dict())

    # save model?
    if config_dict["training_parameters"]["save_models"]:
        PATH = f"{PATH_MODELS}/model_{fold}"
        torch.save(model.state_dict(), PATH)

    all_models.append(model)
    all_best_epochs.append(trainer.best_iteration)

    # Validate
    model.eval()
    with torch.no_grad():
        pred = model(val_dataloader.dataset.arrays)
        all_predictions.append(pred[1].numpy())
        all_true.append(val_dataloader.dataset.arrays[3].numpy())
    
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
with open(f"{PATH_MODELS}/best_epochs", "wb") as fp:   # Pickling best epochs
    pickle.dump(all_best_epochs, fp)
with open(f"{PATH_MODELS}/all_train_losses", "wb") as fp:   # Pickling train losses
    pickle.dump(all_train_losses, fp)
with open(f"{PATH_MODELS}/all_val_losses", "wb") as fp:   # Pickling val losses
    pickle.dump(all_val_losses, fp)
with open(f"{PATH_MODELS}/all_scalers", "wb") as fp:   # Pickling scalers
    pickle.dump(all_scalers, fp)

print("\n ---------------------------------------")
print(" ---------- Script finished ----------")
print(" ---------------------------------------")
