import configparser
import re


def read_config(path_to_config="./config.ini"):
    # Create a ConfigParser object
    config = configparser.ConfigParser()

    # Read the configuration file
    config.read(path_to_config)
    print(config.sections())

    config_dict = dict()

    # data files
    file_names = dict(config.items('file_names'))
    config_dict['file_names'] = file_names

    # data array - names and features
    data_arrays = {}
    # array names
    array_names = re.split(r'[;,\s]+', config.get('arrays', 'array_names'))
    # array columns
    for array in array_names:
        features = re.split(r'[;,\s]+', config.get('arrays', f"col_{array}"))
        if len(features) == 1 and features[0] == 'all':
            features = features[0]
        data_arrays[array] = features
    config_dict['data_arrays'] = data_arrays
    
    # other column_names
    common_columns = dict(config.items('common_columns'))
    config_dict['common_columns'] = common_columns

    # features to preprocess
    # one list per array
    preprocess_arrays = {}
    for array in array_names:
        features = re.split(r'[;,\s]+', config.get('preprocess_arrays', f"{array}"))
        if len(features) == 1 and features[0] == 'all':
            features = features[0]
        preprocess_arrays[array] = features
    config_dict['preprocess'] = preprocess_arrays

    # training_parameters
    config_dict['training_parameters'] = {}
    config_dict['training_parameters']['save_models'] = config.getboolean('training_parameters', 'save_models')
    config_dict['training_parameters']['n_folds'] = config.getint('training_parameters', 'n_folds')
    config_dict['training_parameters']['batch_size'] = config.getint('training_parameters', 'batch_size')
    try:
        config_dict['training_parameters']['batch_size_val'] = config.getint('training_parameters', 'batch_size_val')
    except ValueError:
        config_dict['training_parameters']['batch_size_val'] = None
    config_dict['training_parameters']['batch_size_val']
    config_dict['training_parameters']['num_epochs'] = config.getint('training_parameters', 'num_epochs')
    config_dict['training_parameters']['device'] = config.get('training_parameters', 'device')
    # torch.device("cpu")

    # model_params
    config_dict['model_params'] = {}
    config_dict['model_params']['vae_metabolomics_latent_dim'] = config.getint('model_params', 'vae_metabolomics_latent_dim')
    config_dict['model_params']['vae_genomics_latent_dim'] = config.getint('model_params', 'vae_genomics_latent_dim')
    config_dict['model_params']['transformer_input_dim'] = config.getint('model_params', 'transformer_input_dim')
    config_dict['model_params']['n_heads'] = config.getint('model_params', 'n_heads')
    config_dict['model_params']['transformer_dim_feedforward'] = config_dict['model_params']['transformer_input_dim'] * config_dict['model_params']['n_heads']
    
    return config_dict


if __name__ == "__main__":
    # Call the function to read the configuration file
    config_dict = read_config(path_to_config="./config.ini")
    print(config_dict)
