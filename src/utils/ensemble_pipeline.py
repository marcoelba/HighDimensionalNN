# Ensemble models
import pickle

import torch
import numpy as np


class EnsemblePipeline:
    def __init__(self, model_class, config_dict):
        self.config_dict = config_dict
        self.path_results = config_dict["script_parameters"]["results_folder"]
        self.n_folds = config_dict["training_parameters"]["n_folds"]
        self.device = torch.device(config_dict["training_parameters"]["device"])

        # Load scalers
        try:
            self.scalers = load_scalers()
        except Exception as e:
            self.scalers = None
            print(f"Pickle scalers not loaded: {e}")
        # model dimension parameters
        try:
            self.model_parameters = load_model_paramerers()
        except Exception as e:
            self.model_parameters = None
            print(f"Pickle model_parameters not loaded: {e}")

        # Load torch models
        self.all_models = []
        for fold in range(self.n_folds):
            path = f"{self.path_results}/model_{fold}"

            model = model_class(
                input_dim_genes=self.model_parameters["p_gene"],
                input_dim_metab=self.model_parameters["p_metab"],
                input_patient_features_dim=self.model_parameters["p_static"],
                n_timepoints=self.model_parameters["n_timepoints"],
                model_config=self.config_dict["model_params"]
            ).to(self.device)
            model.load_state_dict(torch.load(path))

            self.all_models.append(model)
        print(f"Fold models loaded")
        self.torch_models = torch.nn.ModuleList(self.all_models)

    def load_scalers(self):
        with open(os.path.join(self.path_results, "all_scalers"), "rb") as fp:
            x = pickle.load(fp)
        return x

    def load_model_paramerers(self):
        with open(os.path.join(self.path_results, "model_parameters"), "rb") as fp:
            x = pickle.load(fp)
        return x

    # def load_shap_values(self):
    #     with open(os.path.join(self.path_results, "all_shap_values"), "rb") as fp:
    #         x = pickle.load(fp)
    #     return x

    def predict_fold(self, x, fold):
        """
        Return a tensor with predictions for one specific fold
        """
        model_fold = self.all_models[fold]
        model_fold.eval()
        with torch.no_grad():
            y_pred = model_fold(x)[-1]
        
        return y_pred

    def predict(self, x_list):
        """
        Return a tensor with predictions on all folds
        """
        all_predictions = []
        for fold in range(self.n_folds):
            y_pred = self.predict_fold(x_list[fold], fold)
            all_predictions.append(y_pred)
        
        return torch.stack(all_predictions, axis=0)

    def predict_mean(self, x):
        """
        Return a tensor with aggregated predictions over folds (axis 0). Default to the mean
        """
        predictions_per_fold = self.predict(x)
        
        return predictions_per_fold.mean(axis=0)


class EnsembleModelSingleTime(torch.nn.Module):
    def __init__(self, full_model, config_dict, model_parameters, all_scalers=None):
        super(EnsembleModelSingleTime, self).__init__()
        self.n_folds = config_dict["training_parameters"]["n_folds"]
        self.config_dict = config_dict
        self.device = torch.device(config_dict["training_parameters"]["device"])
        self.full_model = full_model

        if all_scalers is not None:
            self.torch_scalers_outcome = [TorchScaler(all_scalers[fold]['y_target'][0]) for fold in range(self.n_folds)]
        else:
            self.torch_scalers_outcome = None
        self.torch_models = None
        self.time_to_explain = None
        self.n_timepoints = None

        # Load torch models
        path_results = self.config_dict["script_parameters"]["results_folder"]
        self.n_timepoints = model_parameters["n_timepoints"]
        
        all_models = []
        for fold in range(self.n_folds):
            path = f"{path_results}/model_{fold}"

            model = full_model(
                input_dim_genes=model_parameters["p_gene"],
                input_dim_metab=model_parameters["p_metab"],
                input_patient_features_dim=model_parameters["p_static"],
                n_timepoints=model_parameters["n_timepoints"],
                model_config=self.config_dict["model_params"]
            ).to(self.device)
            model.load_state_dict(torch.load(path))

            all_models.append(model)
        print(f"Fold models loaded")
        self.torch_models = torch.nn.ModuleList(all_models)
    
    def forward(self, *x):
        x_list = list(x)
        all_outputs = []
        for fold, model in enumerate(self.torch_models):
            model.eval()
            output = model(x_list)[2][:, [self.time_to_explain]]
            # transform output back to original scale
            if self.torch_scalers_outcome is not None:
                output = self.torch_scalers_outcome[fold].inverse_transform(output)
                output = torch.exp(output)
            all_outputs.append(output)
        return torch.stack(all_outputs).mean(dim=0)


class TorchScaler:
    def __init__(self, scaler, dtype=torch.float32):
        self.mean = torch.tensor(scaler.mean_, dtype=dtype)
        self.scale = torch.tensor(scaler.scale_, dtype=dtype)

    def transform(self, x):
        """
            x: torch.Tensor
        """
        return (x - self.mean) / self.scale

    def inverse_transform(self, x):
        """
            x: torch.Tensor
        """
        return x * self.scale + self.mean
