# Pre-Processing
import numpy as np
import copy

from sklearn.preprocessing import StandardScaler


def preprocess_train(dict_train: dict, dict_val: dict, config_dict: dict):
    """
    Preprocess arrays
    
    Parameters
    ----------
    list_train : dict of ndarray
        Training array of shape (n_samples, n_meals, n_features)
    list_val : dict of ndarray
        Validation array of shape (n_samples, n_meals, n_features)
    config_dict : dict
        Config Dictionary
    """
    
    features_to_preprocess = config_dict["preprocess"]
    data_arrays = config_dict["data_arrays"]

    all_arrays = dict_train.keys()
    # Initialize scalers
    scalers = dict()
    dict_train_preproc = dict()
    dict_val_preproc = dict()

    for array in all_arrays:

        train_array = dict_train[array]
        val_array = dict_val[array]
        n_features = train_array.shape[-1]

        if array in features_to_preprocess:
            scalers[array] = dict()

            if features_to_preprocess[array] == 'all':
                features_to_preprocess_idx = [jj for jj in range(n_features)]
            else:
                features_to_preprocess_idx = [jj for jj in range(n_features) if data_arrays[array][jj] in features_to_preprocess[array]]
            
            # make arrays flat
            train_array_flat = train_array.reshape(-1, n_features)
            val_array_flat = val_array.reshape(-1, n_features)

            # Process features
            for idx in features_to_preprocess_idx:
                # Extract the feature column
                train_feature = copy.deepcopy(train_array_flat[:, idx])
                val_feature = copy.deepcopy(val_array_flat[:, idx])

                # Remove NaN values for fitting
                train_feature_no_nan = train_feature[~np.isnan(train_feature)]
                val_feature_no_nan = val_feature[~np.isnan(val_feature)]
                
                if len(train_feature_no_nan) > 0:
                    # Fit scaler on training data
                    scaler = StandardScaler()
                    scaler.fit(train_feature_no_nan.reshape(-1, 1))
                    
                    # Transform both train and validation
                    train_array_flat[~np.isnan(train_feature), idx] = scaler.transform(
                        train_feature_no_nan.reshape(-1, 1)
                    ).flatten()
                    
                    val_array_flat[~np.isnan(val_feature), idx] = scaler.transform(
                        val_feature_no_nan.reshape(-1, 1)
                    ).flatten()
                    
                    scalers[array][idx] = scaler
        
            # Reshape back to original shape
            train_processed = train_array_flat.reshape(train_array.shape)
            val_processed = val_array_flat.reshape(val_array.shape)
        
            dict_train_preproc[array] = train_processed
            dict_val_preproc[array] = val_processed

        else:
            dict_train_preproc[array] = train_array
            dict_val_preproc[array] = val_array
    
    return dict_train_preproc, dict_val_preproc, scalers


def preprocess_transform(dict_new_data: dict, scalers: dict, config_dict: dict, verbose=False):
    """
    Preprocess new data using the already trained scalers
    
    Parameters
    ----------
    dict_new_data : dict of ndarray
        Training array of shape (n_samples, n_meals, n_features)
    scalers : dictionary of scalers
        Trained scalers
    config_dict : dict
        config Dictionary
    """

    features_to_preprocess = config_dict["preprocess"]
    data_arrays = config_dict["data_arrays"]

    all_arrays = dict_new_data.keys()
    dict_test_preproc = dict()

    for array in all_arrays:

        if verbose:
            print(f"\n ------------ Processing feature {array} ------------")
        
        test_array = dict_new_data[array]
        n_features = test_array.shape[-1]

        if array in features_to_preprocess:

            if features_to_preprocess[array] == 'all':
                features_to_preprocess_idx = [jj for jj in range(n_features)]
            else:
                features_to_preprocess_idx = [jj for jj in range(n_features) if data_arrays[array][jj] in features_to_preprocess[array]]

            # make arrays flat
            test_array_flat = test_array.reshape(-1, n_features)
            test_array_flat.shape

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
                    test_array_flat[~np.isnan(feature), idx] = scalers[array][idx].transform(
                        feature_no_nan.reshape(-1, 1)
                    ).flatten()
            
            # Reshape back to original shape
            test_processed = test_array_flat.reshape(test_array.shape)
            dict_test_preproc[array] = test_processed
            if verbose:
                print(f"NAs in feature AFTER processing: {np.isnan(test_array_flat).sum()}")

        else:
            dict_test_preproc[array] = test_array
    
    return dict_test_preproc
