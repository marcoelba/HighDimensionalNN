# Pre-Processing
import numpy as np
import copy

from sklearn.preprocessing import StandardScaler


class Preprocessing:
    def __init__(self, config_dict: dict):
        self.config_dict = config_dict
        self.features_to_preprocess = config_dict["preprocess"]
        self.data_arrays = config_dict["data_arrays"]

        self.scalers = dict()
    
    def _get_features_indeces(self, array_name, n_features):
        if self.features_to_preprocess[array_name] == 'all':
            features_to_preprocess_idx = [jj for jj in range(n_features)]
        else:
            features_to_preprocess_idx = [jj for jj in range(n_features) if self.data_arrays[array_name][jj] in self.features_to_preprocess[array_name]]
        
        return features_to_preprocess_idx

    def train(self, dict_train: dict):
        """
        Preprocess of longitudinal arrays
        
        Parameters
        ----------
        dict_train : dict of ndarray
            Training array of shape (n_samples, n_meals, n_features)
        """
        all_arrays = dict_train.keys()

        for array in all_arrays:

            train_array = dict_train[array]
            n_features = train_array.shape[-1]

            if array in self.features_to_preprocess:
                self.scalers[array] = dict()
                features_to_preprocess_idx = self._get_features_indeces(array, n_features)
                
                # make arrays flat
                train_array_flat = train_array.reshape(-1, n_features)

                # Process features
                for idx in features_to_preprocess_idx:
                    # Extract the feature column
                    train_feature = copy.deepcopy(train_array_flat[:, idx])

                    # Remove NaN values for fitting
                    train_feature_no_nan = train_feature[~np.isnan(train_feature)]
                    
                    if len(train_feature_no_nan) > 0:
                        # Fit scaler on training data
                        scaler = StandardScaler()
                        scaler.fit(train_feature_no_nan.reshape(-1, 1))
                        self.scalers[array][idx] = scaler
        
    def transform(self, dict_arrays: dict, verbose=False):
        """
        Preprocess new longitudinal data using the already trained scalers
        
        Parameters
        ----------
        dict_arrays : dict of ndarray
            Training array of shape (n_samples, n_meals, n_features)
        """

        all_arrays = dict_arrays.keys()
        dict_arrays_preproc = dict()

        for array in all_arrays:

            if verbose:
                print(f"\n ------------ Processing feature {array} ------------")
            
            test_array = dict_arrays[array]
            n_features = test_array.shape[-1]

            if array in self.features_to_preprocess:

                features_to_preprocess_idx = self._get_features_indeces(array, n_features)

                # make arrays flat
                test_array_flat = test_array.reshape(-1, n_features)

                if verbose:
                    print(f"NAs in feature: {np.isnan(test_array_flat).sum()}")

                # Process features
                for idx in features_to_preprocess_idx:

                    # Extract the feature column
                    feature = copy.deepcopy(test_array_flat[:, idx])
                    # Remove NaN values for fitting
                    feature_no_nan = feature[~np.isnan(feature)]
                    
                    if len(feature_no_nan) > 0:
                        # Transform validation
                        test_array_flat[~np.isnan(feature), idx] = self.scalers[array][idx].transform(
                            feature_no_nan.reshape(-1, 1)
                        ).flatten()
                
                # Reshape back to original shape
                test_processed = test_array_flat.reshape(test_array.shape)
                dict_arrays_preproc[array] = test_processed
                if verbose:
                    print(f"NAs in feature AFTER processing: {np.isnan(test_array_flat).sum()}")

            else:
                dict_arrays_preproc[array] = test_array
        
        return dict_arrays_preproc
