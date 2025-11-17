import copy
import torch

from src.utils.features_preprocessing import preprocess_transform
from src.utils import data_loading_wrappers


def prepare_data_for_shap(dict_shap, all_scalers, config_dict, patient_index=None, verbose=False):

    DEVICE = torch.device(config_dict["training_parameters"]["device"])
    FEATURES_KEYS = list(config_dict["preprocess"].keys())[:-1]
    N_FOLDS = config_dict["training_parameters"]["n_folds"]
    
    features_label_per_folds = []

    for fold in range(N_FOLDS):
        # apply feature preprocessing
        dict_arrays_preproc = preprocess_transform(
            copy.deepcopy(dict_shap), all_scalers[fold], config_dict, verbose=verbose
        )
        if dict_arrays_preproc["y_baseline"].shape[-1] == 1:
            dict_arrays_preproc["y_baseline"] = dict_arrays_preproc["y_baseline"][..., 0]
            dict_arrays_preproc["y_target"] = dict_arrays_preproc["y_target"][..., 0]

        if patient_index is not None:
            tensor_data = [
                torch.FloatTensor(array[patient_index]).to(DEVICE) for key, array in dict_arrays_preproc.items()
            ]
        else:
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
        # append
        features_label_per_folds.append(tensor_input)
    
    features_combined = []
    for feature in range(len(FEATURES_KEYS)):
        tensor_feature = []
        for fold in range(N_FOLDS):
            tensor_feature.append(features_label_per_folds[fold][feature])
        features_combined.append(torch.cat(tensor_feature, dim=0))
    
    return features_combined, features_label_per_folds
