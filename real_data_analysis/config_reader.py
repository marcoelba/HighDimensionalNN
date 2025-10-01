import configparser


def read_config(path_to_config="./config.ini"):
    # Create a ConfigParser object
    config = configparser.ConfigParser()

    # Read the configuration file
    config.read(path_to_config)
    print(config.sections())

    config_dict = dict()

    # data files
    config_dict['file_names'] = {}
    config_dict['file_names']['path_to_features_data'] = config.get('file_names', 'path_to_features_data')
    config_dict['file_names']['path_to_clinical_data'] = config.get('file_names', 'path_to_clinical_data')
    config_dict['file_names']['path_to_gene_names'] = config.get('file_names', 'path_to_gene_names')

    # array names
    config_dict['array_names'] = {}
    config_dict['array_names']['genes'] = config.get('array_names', 'genes')
    config_dict['array_names']['static_patient_features'] = config.get('array_names', 'static_patient_features')
    config_dict['array_names']['y_baseline'] = config.get('array_names', 'y_baseline')
    config_dict['array_names']['y_target'] = config.get('array_names', 'y_target')

    # column_names
    config_dict['column_names'] = {}
    config_dict['column_names']['patient_id'] = config.get('column_names', 'patient_id')
    config_dict['column_names']['patient_meal_id'] = config.get('column_names', 'patient_meal_id')
    config_dict['column_names']['col_meal'] = config.get('column_names', 'col_meal')
    config_dict['column_names']['col_visit'] = config.get('column_names', 'col_visit')
    config_dict['column_names']['col_time'] = config.get('column_names', 'col_time')
    config_dict['column_names']['col_outcome'] = config.get('column_names', 'col_outcome')
    config_dict['column_names']['col_sex'] = config.get('column_names', 'col_sex')
    config_dict['column_names']['col_age'] = config.get('column_names', 'col_age')
    config_dict['column_names']['col_bmi'] = config.get('column_names', 'col_bmi')
    # make additional parameters
    config_dict['column_names']['patient_cols'] = [
        config_dict['column_names']['col_sex'],
        config_dict['column_names']['col_age'],
        config_dict['column_names']['col_bmi']
    ]

    # features to preprocess
    # one list per array
    config_dict['preprocess'] = {}
    config_dict['preprocess'][config_dict['array_names']['genes']] = "all"
    config_dict['preprocess'][config_dict['array_names']['static_patient_features']] = [
        config_dict['column_names']['col_age'],
        config_dict['column_names']['col_bmi']
    ]
    config_dict['preprocess'][config_dict['array_names']['y_baseline']] = config_dict['column_names']['col_outcome']
    config_dict['preprocess'][config_dict['array_names']['y_target']] = config_dict['column_names']['col_outcome']
    
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
    config_dict['model_params']['latent_dim'] = config.getint('model_params', 'latent_dim')
    config_dict['model_params']['transformer_input_dim'] = config.getint('model_params', 'transformer_input_dim')
    config_dict['model_params']['n_heads'] = config.getint('model_params', 'n_heads')
    config_dict['model_params']['transformer_dim_feedforward'] = config_dict['model_params']['transformer_input_dim'] * config_dict['model_params']['n_heads']
    
    return config_dict


if __name__ == "__main__":
    # Call the function to read the configuration file
    config_dict = read_config(path_to_config="./config.ini")
    print(config_dict)
