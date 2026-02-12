# Ensemble models
import pickle
import copy
import os

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.utils.data_handling import train_data_batching
from src.utils import training_wrapper
from src.utils.files_io_helpers import save_pickle, load_pickle


class EnsemblePipeline:
    def __init__(self, model_class, preprocessing_class, config_dict, model_dimension_definition=None):
        self.config_dict = config_dict
        self.path_results = config_dict["script_parameters"]["results_folder"]
        self.n_folds = config_dict["training_parameters"]["n_folds"]
        self.device = torch.device(config_dict["training_parameters"]["device"])
        self.model_class = model_class
        self.preprocessing_class = preprocessing_class
        
        self.model_dimension_definition = model_dimension_definition
        self.all_models = []

        # Load scalers
        try:
            self.all_scalers = self.load_scalers()
        except Exception as e:
            self.all_scalers = []
            print(f"Pickle scalers not loaded: {e}")
        
        # model dimension parameters
        try:
            self.model_dimension_definition = self.load_model_dim_parameters()
        except Exception as e:
            print(f"Pickle model_dimension_definition not loaded: {e}")
    
    def _trace_plot(self, trainer, plot_name):
        fig = plt.figure()
        plt.plot(trainer.losses["train"], label="Train")
        plt.plot(trainer.losses["val"], label="Val")
        plt.legend()
        fig.savefig(os.path.join(self.path_results, plot_name), format="pdf")
        plt.close()
            
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
        
        x = load_pickle(
                os.path.join(self.path_results, self.config_dict["saving_file_names"]["pickle_all_scalers"])
        )
        return x

    def load_model_dim_parameters(self):
        x = load_pickle(
            os.path.join(self.path_results, self.config_dict["saving_file_names"]["pickle_model_dimension_definition"])
        )
        return x
    
    def train(self, dict_arrays, feature_names, reduce_on_plateau=False):
        # save current model init definition
        if self.config_dict["training_parameters"]["save_models"]:
            save_pickle(
                self.model_dimension_definition,
                os.path.join(
                    self.path_results,
                    self.config_dict["saving_file_names"]["pickle_model_dimension_definition"]
                )
            )
        
        n_individuals = dict_arrays["y_target"].shape[0]
        train_indices = np.random.permutation(np.arange(0, n_individuals))
        # Split into k folds
        folds = np.array_split(train_indices, self.n_folds)
        
        # to store results
        all_train_losses = []
        all_val_losses = []
        predictions_val_folds = []
        ground_truth_val_folds = []
        
        for fold in range(self.n_folds):
            print("\n -----------------------------------------------------------")
            print(f"Running k-fold validation on fold {fold+1} of {self.n_folds}")

            # mask current fold for use in validation
            train_mask = np.ones(n_individuals, dtype=int)
            train_mask[folds[fold]] = False

            # Split
            dict_train = {name: arr[train_mask == 1] for name, arr in dict_arrays.items()}
            dict_val = {name: arr[train_mask == 0] for name, arr in dict_arrays.items()}

            # train and apply feature preprocessing
            features_preprocessing = self.preprocessing_class(self.config_dict, feature_names)
            features_preprocessing.train(dict_train)
            self.all_scalers.append(features_preprocessing)

            dict_train_preproc = features_preprocessing.transform(dict_train)
            dict_val_preproc = features_preprocessing.transform(dict_val)

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
            trainer = training_wrapper.Training(
                train_dataloader,
                val_dataloader,
                reduce_on_plateau=reduce_on_plateau,
                noisy_gradient=False
            )
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

            # Run predictions only on the validation set of the k-fold
            model.eval()
            with torch.no_grad():
                pred = model(val_dataloader.dataset.arrays)
                predictions_val_folds.append(pred[-1].numpy())
                ground_truth_val_folds.append(val_dataloader.dataset.arrays[-1].numpy())
                print(f"RMSE fold {fold}: {np.sqrt(np.mean((pred[-1].numpy() - val_dataloader.dataset.arrays[-1].numpy())**2))}")

            print(f"train loss: {np.min(trainer.losses['train'])}")
            print(f"val loss: {np.min(trainer.losses['val'])}")
            all_train_losses.append(np.min(trainer.losses['train']))
            all_val_losses.append(np.min(trainer.losses['val']))

            # if saving loss traces
            if self.config_dict["training_parameters"]["save_models"]:
                plot_name = f"train_val_loss_fold_{fold}.pdf"
                self._trace_plot(trainer, plot_name)
        # END k-fold training

        # saving training results
        if self.config_dict["training_parameters"]["save_models"]:
            save_pickle(
                self.all_scalers,
                os.path.join(self.path_results, self.config_dict["saving_file_names"]["pickle_all_scalers"])
            )
            save_pickle(
                all_train_losses,
                os.path.join(self.path_results, self.config_dict["saving_file_names"]["pickle_all_train_losses"])
            )
            save_pickle(
                all_val_losses,
                os.path.join(self.path_results, self.config_dict["saving_file_names"]["pickle_all_val_losses"])
            )
            save_pickle(
                predictions_val_folds,
                os.path.join(self.path_results, self.config_dict["saving_file_names"]["pickle_predictions_val"])
            )
            save_pickle(
                ground_truth_val_folds,
                os.path.join(self.path_results, self.config_dict["saving_file_names"]["pickle_ground_truth_val"])
            )
            print("\nPickles saved successfully")

    def predict_fold(self, x, fold):
        """
        Return a tensor with predictions for one specific fold
        """
        model_fold = self.all_models[fold]
        model_fold.eval()
        with torch.no_grad():
            y_pred = model_fold(x)[-1]
        
        return y_pred

    def predict(self, dict_arrays) -> list:
        """
        Return a tensor with predictions on all folds
        """
        all_predictions = []
        all_predictions_original = []
        for fold in range(self.n_folds):
            features_preprocessing = self.all_scalers[fold]
            dict_preproc = features_preprocessing.transform(dict_arrays)

            # remove last dimension for outcome with only one dimension
            if dict_preproc["y_target"].shape[-1] == 1:
                dict_preproc["y_target"] = dict_preproc["y_target"][..., 0]
                dict_preproc["y_baseline"] = dict_preproc["y_baseline"][..., 0]

            # get tensors
            tensor_data = [torch.FloatTensor(array).to(self.device) for key, array in dict_preproc.items()]
            y_pred = self.predict_fold(tensor_data, fold)
            y_pred_original = features_preprocessing.scalers["y_target"][0].inverse_transform(y_pred.reshape(-1, y_pred.shape[-1]))
            y_pred_original = y_pred_original.reshape(y_pred.shape)

            all_predictions.append(y_pred.numpy())
            all_predictions_original.append(y_pred_original)
        return all_predictions, all_predictions_original
