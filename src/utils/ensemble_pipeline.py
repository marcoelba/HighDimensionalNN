# Ensemble models
import pickle
import copy
import os

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.utils.data_handling import train_data_batching
from src.utils import training_wrapper


class EnsemblePipeline:
    def __init__(self, model_class, features_preprocessing, config_dict, model_dimension_definition=None):
        self.config_dict = config_dict
        self.path_results = config_dict["script_parameters"]["results_folder"]
        self.n_folds = config_dict["training_parameters"]["n_folds"]
        self.device = torch.device(config_dict["training_parameters"]["device"])
        self.model_class = model_class
        self.features_preprocessing = features_preprocessing
        
        self.model_dimension_definition = model_dimension_definition
        self.all_models = []

        # Load scalers
        try:
            self.all_scalers = load_scalers()
        except Exception as e:
            self.all_scalers = []
            print(f"Pickle scalers not loaded: {e}")
        
        # model dimension parameters
        try:
            self.model_dimension_definition = load_model_dim_parameters()
        except Exception as e:
            print(f"Pickle model_dimension_definition not loaded: {e}")
    
    def __trace_plot(self, trainer, plot_name):
        fig = plt.figure()
        plt.plot(trainer.losses["train"], label="Train")
        plt.plot(trainer.losses["val"], label="Val")
        plt.legend()
        fig.savefig(os.path.join(self.path_results, plot_name), format="pdf")
        plt.close()
    
    def __save_pickle(self, file, name_file):
        with open(os.path.join(self.path_results, name_file), "wb") as fp:
            pickle.dump(file, fp)

    def __load_pickle(self, name_file):
        with open(os.path.join(self.path_results, name_file), "rb") as fp:
            x = pickle.load(fp)
        return x
        
    def load_trained_models(self):

        for fold in range(self.n_folds):
            path = os.path.join(self.path_results, f"model_{fold}")
            model = self.model_class(
                self.config_dict["model_params"],
                self.model_dimension_definition
            ).to(self.device)
            model.load_state_dict(torch.load(path))
            self.all_models.append(model)
        print(f"Fold models loaded")
        self.torch_models = torch.nn.ModuleList(self.all_models)

    def load_scalers(self):
        x = self.__load_pickle(
            os.path.join(self.path_results, self.config_dict["saving_file_names"]["pickle_all_scalers"])
        )
        return x

    def load_model_dim_parameters(self):
        x = self.__load_pickle(
            os.path.join(self.path_results, self.config_dict["file_names"]["pickle_model_dimension_definition"])
        )
        return x
    
    def train(self, dict_arrays):
        # save current model init definition
        if self.config_dict["training_parameters"]["save_models"]:
            self.__save_pickle(
                self.model_dimension_definition,
                self.config_dict["saving_file_names"]["pickle_model_dimension_definition"]
            )
        
        n_individuals = dict_arrays["y_target"].shape[0]
        train_indices = np.random.permutation(np.arange(0, n_individuals))
        # Split into k folds
        folds = np.array_split(train_indices, self.n_folds)
        self.all_scalers = []
        self.all_train_losses = []
        self.all_val_losses = []
        
        for fold in range(self.n_folds):
            print(f"Running k-fold validation on fold {fold+1} of {self.n_folds}")

            # mask current fold for use in validation
            train_mask = np.ones(n_individuals, dtype=int)
            train_mask[folds[fold]] = False

            # Split
            dict_train = {name: arr[train_mask == 1] for name, arr in dict_arrays.items()}
            dict_val = {name: arr[train_mask == 0] for name, arr in dict_arrays.items()}

            # train and apply feature preprocessing
            self.features_preprocessing.train(dict_train)
            self.all_scalers.append(self.features_preprocessing)

            dict_train_preproc = self.features_preprocessing.transform(dict_train)
            dict_val_preproc = self.features_preprocessing.transform(dict_val)

            # remove last dimension for outcome with only one dimension
            if dict_train_preproc["y_target"].shape[-1] == 1:
                dict_train_preproc["y_target"] = dict_train_preproc["y_target"][..., 0]
                dict_val_preproc["y_target"] = dict_val_preproc["y_target"][..., 0]
                dict_train_preproc["y_baseline"] = dict_train_preproc["y_baseline"][..., 0]
                dict_val_preproc["y_baseline"] = dict_val_preproc["y_baseline"][..., 0]

            # get tensors
            tensor_data_train = [torch.FloatTensor(array).to(self.device) for key, array in dict_train_preproc.items()]
            tensor_data_val = [torch.FloatTensor(array).to(self.device) for key, array in dict_val_preproc.items()]

            # Validation batch size - just do one
            y_val_shape = dict_val_preproc["y_target"].shape
            batch_size_val = y_val_shape[0] * y_val_shape[1]
            # Train batch size
            y_train_shape = dict_train_preproc["y_target"].shape

            # data loaders
            train_dataloader = train_data_batching.make_data_loader(
                *tensor_data_train,
                batch_size=self.config_dict["training_parameters"]["batch_size"],
                feature_dimensions=-1,
                reshape=True,
                drop_missing=True
            )

            val_dataloader = train_data_batching.make_data_loader(
                *tensor_data_val,
                batch_size=batch_size_val,
                feature_dimensions=-1,
                reshape=True,
                drop_missing=True
            )

            # ---------------------- Model Setup ----------------------
            model = self.model_class(
                self.config_dict["model_params"],
                self.model_dimension_definition
            ).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            # Training Loop
            trainer = training_wrapper.Training(train_dataloader, val_dataloader, noisy_gradient=False)
            trainer.training_loop(
                model,
                optimizer,
                self.config_dict["training_parameters"]["num_epochs"],
                gradient_noise_std=0.0
            )

            # use the model at the best validation iteration
            model.load_state_dict(trainer.best_model.state_dict())

            # save model?
            if self.config_dict["training_parameters"]["save_models"]:
                path = os.path.join(self.path_results, f"model_{fold}")
                torch.save(model.state_dict(), path)

            self.all_models.append(model)

            # Validate
            # model.eval()
            # with torch.no_grad():
            #     pred = model(val_dataloader.dataset.arrays)
            #     all_predictions.append(pred[-1].numpy())
            #     all_true.append(val_dataloader.dataset.arrays[-1].numpy())

            self.all_train_losses.append(np.min(trainer.losses["train"]))
            self.all_val_losses.append(np.min(trainer.losses["val"]))

            # if saving loss traces
            if self.config_dict["training_parameters"]["save_models"]:
                plot_name = f"train_val_loss_fold_{fold}.pdf"
                self.__trace_plot(trainer, plot_name)
        
        # saving training results
        if self.config_dict["training_parameters"]["save_models"]:
            self.__save_pickle(self.all_scalers, self.config_dict["saving_file_names"]["pickle_all_scalers"])
            self.__save_pickle(self.all_train_losses, self.config_dict["saving_file_names"]["pickle_all_train_losses"])
            self.__save_pickle(self.all_val_losses, self.config_dict["saving_file_names"]["pickle_all_val_losses"])

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
