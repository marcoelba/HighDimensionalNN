import numpy as np
import torch
import pickle
import os

from real_data_analysis.features_preprocessing import preprocess, preprocess_transform
from real_data_analysis.config_reader import read_config

from src.vae_attention.full_model import DeltaTimeAttentionVAE


def ensemble_predict(x):
    # This list will hold the predictions from each (model, scaler) pair
    all_predictions = []

    for fold in range(N_FOLDS):
        print(f"Running k-fold validation on fold {fold+1} of {N_FOLDS}")
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
        # fold-model prediction
        model = all_models[fold]
        model.eval()
        with torch.no_grad():
            prediction = model(tensor_input)
            all_predictions.append(prediction[1].numpy())  # here only y_pred

    # Stack predictions and compute mean along the ensemble axis
    predictions = np.array(all_predictions)
    ensemble_mean = np.mean(predictions, axis=0)

    return ensemble_mean
