# Pre-Processing
import numpy as np
from sklearn.preprocessing import StandardScaler


def preprocess(dict_train: dict, dict_val: dict, features_to_preprocess: dict):
    """
    Preprocess arrays
    
    Parameters
    ----------
    list_train : dict of ndarray
        Training array of shape (n_samples, n_meals, n_features)
    list_val : dict of ndarray
        Validation array of shape (n_samples, n_meals, n_features)
    features_to_preprocess : dict
        Dictionary with array name as key and list of features as value
    """
    
    all_arrays_keys = dict_train.keys()
    # Initialize scalers
    scalers = dict()
    dict_train_preproc = dict()
    dict_val_preproc = dict()

    for key in all_arrays_keys:

        train_array = dict_train[key]
        val_array = dict_val[key]

        if key in features_to_preprocess:
            scalers[key] = dict()
            dict_features_index = features_to_preprocess[key]
                        
            n_features = train_array.shape[-1]

            train_array_flat = train_array.reshape(-1, n_features)
            val_array_flat = val_array.reshape(-1, n_features)

            # Process features
            for idx in dict_features_index.values():
                # Extract the feature column
                train_feature = train_array_flat[:, idx]
                val_feature = val_array_flat[:, idx]

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
                    
                    scalers[key][idx] = scaler
        
            # Reshape back to original shape
            train_processed = train_array_flat.reshape(train_array.shape)
            val_processed = val_array_flat.reshape(val_array.shape)
        
            dict_train_preproc[key] = train_processed
            dict_val_preproc[key] = val_processed

        else:
            dict_train_preproc[key] = train_array
            dict_val_preproc[key] = val_array
    
    return dict_train_preproc, dict_val_preproc, scalers


def preprocess_transform(dict_new_data: dict, scalers: dict, features_to_preprocess: dict):
    """
    Preprocess new data using the already trained scalers
    
    Parameters
    ----------
    dict_new_data : dict of ndarray
        Training array of shape (n_samples, n_meals, n_features)
    features_to_preprocess : dict
        Dictionary with array name as key and list of features as value
    """
    
    all_arrays_keys = dict_new_data.keys()
    dict_val_preproc = dict()

    for key in all_arrays_keys:

        val_array = dict_new_data[key]

        if key in features_to_preprocess:

            dict_features_index = features_to_preprocess[key]                        
            n_features = val_array.shape[-1]
            val_array_flat = val_array.reshape(-1, n_features)

            # Process features
            for idx in dict_features_index.values():
                # Extract the feature column
                val_feature = val_array_flat[:, idx]

                # Remove NaN values for fitting
                val_feature_no_nan = val_feature[~np.isnan(val_feature)]
                
                if len(val_feature_no_nan) > 0:                    
                    # Transform validation
                    val_array_flat[~np.isnan(val_feature), idx] = scalers[key][idx].transform(
                        val_feature_no_nan.reshape(-1, 1)
                    ).flatten()
            
            # Reshape back to original shape
            val_processed = val_array_flat.reshape(val_array.shape)
            dict_val_preproc[key] = val_processed

        else:
            dict_val_preproc[key] = val_array
    
    return dict_val_preproc
