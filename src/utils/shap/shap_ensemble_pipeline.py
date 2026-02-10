# Ensemble pipeline for Shap
import copy

import torch
import numpy as np

from src.utils.features_preprocessing import TorchScaler


class ShapEnsembleModelSingleTime(torch.nn.Module):
    def __init__(self, torch_ensemble_model, config_dict, all_scalers=None):
        super(ShapEnsembleModelSingleTime, self).__init__()
        self.n_folds = config_dict["training_parameters"]["n_folds"]
        self.config_dict = config_dict
        self.torch_ensemble_model = torch_ensemble_model

        if all_scalers is not None:
            self.torch_scalers_outcome = [TorchScaler(all_scalers[fold].scalers["y_target"][0]) for fold in range(self.n_folds)]
        else:
            self.torch_scalers_outcome = None
        
        self.time_to_explain = None
    
    def forward(self, *x):
        x_list = list(x)
        all_outputs = []
        for fold, model in enumerate(self.torch_ensemble_model):
            model.eval()
            output = model(x_list)[2][:, [self.time_to_explain]]
            # transform output back to original scale
            if self.torch_scalers_outcome is not None:
                output = self.torch_scalers_outcome[fold].inverse_transform(output)
                output = torch.exp(output)
            all_outputs.append(output)
            # return mean over folds
        return torch.stack(all_outputs).mean(dim=0)



class ShapEnsembleModel(torch.nn.Module):
    def __init__(self, model_pipeline, config_dict):
        super(ShapEnsembleModel, self).__init__()

        self.config_dict = config_dict
        self.path_results = config_dict["script_parameters"]["results_folder"]
        self.n_folds = config_dict["training_parameters"]["n_folds"]
        self.device = torch.device(config_dict["training_parameters"]["device"])
        self.model_pipeline = model_pipeline

        # Load scalers
        self.all_scalers = model_pipeline.all_scalers
        assert len(self.all_scalers) > 0
        # load trained models
        self.all_models = model_pipeline.all_models
        assert len(self.all_models) > 0

        if self.all_scalers is not None:
            self.torch_scalers_outcome = [TorchScaler(self.all_scalers[fold].scalers["y_target"][0]) for fold in range(self.n_folds)]
        else:
            self.torch_scalers_outcome = None

        # to be changed dynamically
        self.time_to_explain = None
        self.fold_to_explain = None

    def preprocessing(self, dict_arrays):
        # do the preprocessing for each fold on all data
        preprocessed_features_flat_per_fold = []
        preprocessed_features_per_fold = []
        predictions_per_fold = []
        
        for fold in range(self.n_folds):
            features_preprocessing = self.all_scalers[fold]
            dict_preproc = features_preprocessing.transform(dict_arrays)

            # remove last dimension for outcome with only one dimension
            if dict_preproc["y_target"].shape[-1] == 1:
                dict_preproc["y_target"] = dict_preproc["y_target"][..., 0]
                dict_preproc["y_baseline"] = dict_preproc["y_baseline"][..., 0]
            # get tensors
            tensor_data = [torch.FloatTensor(array).to(self.device) for key, array in dict_preproc.items()]
            preprocessed_features_per_fold.append(tensor_data[::-1])
            # need to flatten over all dimensions, except batch (first) and time (last)
            tensor_data_flat = []
            self.tensor_not_na_indexes = []
            self.features_flat_shape = []
            self.features_flat_shape_no_na = []
            self.features_shape = []
            for feature in tensor_data[:-1]:
                feature_shape = feature.shape
                self.features_shape.append(feature_shape)
                # featurecopy.deepcopy(feature)
                # NAs
                where_not_na = sum_not_nan(feature) > 0
                where_not_na_flat = where_not_na.reshape(-1)
                self.tensor_not_na_indexes.append(where_not_na_flat)
                # flattening
                feature_flat = feature.reshape(-1, feature_shape[-1])
                self.features_flat_shape.append(feature_flat.shape)
                feature_flat_not_na = feature_flat[where_not_na_flat]
                self.features_flat_shape_no_na.append(feature_flat_not_na.shape)
                tensor_data_flat.append(feature_flat_not_na)
            # append features except the target
            preprocessed_features_flat_per_fold.append(tensor_data_flat)

            # predictions
            y_pred = self.model_pipeline.predict_fold(tensor_data, fold)
            y_pred_original = features_preprocessing.scalers["y_target"][0].inverse_transform(y_pred.reshape(-1, y_pred.shape[-1]))
            y_pred_original = np.exp(y_pred_original.reshape(y_pred.shape))
            predictions_per_fold.append(y_pred_original)

        return preprocessed_features_flat_per_fold, preprocessed_features_per_fold, predictions_per_fold

    def forward(self, *x):
        """
            This forward call has to be done for each time and each fold
            This is needed because of the particular way that GradientExplainer works

            Args:
                x: list of Tensors with pre-processed features for each fold
        """
        model = self.all_models[self.fold_to_explain]
        x_list = list(x) # take only the current fold of preprocessed features 
        model.eval()
        output = model(x_list)[2][..., [self.time_to_explain]]
        # transform output back to original scale
        if self.torch_scalers_outcome is not None:
            output = self.torch_scalers_outcome[self.fold_to_explain].inverse_transform(output)
            output = torch.exp(output)
        return output


def sum_not_nan(x):
    x_shape = x.shape
    return (~torch.isnan(x)).sum(axis=-1)
