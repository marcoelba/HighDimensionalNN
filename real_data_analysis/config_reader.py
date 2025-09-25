import configparser


def read_config(path_to_config="./config.ini"):
    # Create a ConfigParser object
    config = configparser.ConfigParser()

    # Read the configuration file
    config.read(path_to_config)

    # data files
    path_to_features_data = config.get('file_names', 'path_to_features_data')
    path_to_clinical_data = config.get('file_names', 'path_to_clinical_data')
    path_to_gene_names = config.get('file_names', 'path_to_gene_names')

    # array names
    genes_array = config.get('array_names', 'genes')
    patient_features_array = config.get('array_names', 'patient_features')
    y_baseline_array = config.get('array_names', 'y_baseline')
    y_target_array = config.get('array_names', 'y_target')

    # column_names
    patient_id = config.get('column_names', 'patient_id')
    patient_meal_id = config.get('column_names', 'patient_meal_id')
    col_meal = config.get('column_names', 'col_meal')
    patient_id = config.get('column_names', 'patient_id')
    patient_id = config.get('column_names', 'patient_id')

[column_names]
patient_meal_id = "ID_Meal"
col_meal = "Meal"
col_visit = "Visit"
col_time = "Time"
col_outcome = "TG"
col_sex = "Sex"
col_age = "Age"
col_bmi = "BMI"
# patient_cols = [COL_SEX, COL_AGE, COL_BMI]

[training_parameters]
save_models = True
n_folds = 2
batch_size = 50
batch_size_val = None
num_epochs = 10
device = torch.device("cpu")


[model_params]
latent_dim = 10
transformer_input_dim = 256
n_heads = 4
#transformer_dim_feedforward = transformer_input_dim * 4

    debug_mode = config.getboolean('General', 'debug')
    log_level = config.get('General', 'log_level')
    db_name = config.get('Database', 'db_name')
    db_host = config.get('Database', 'db_host')
    db_port = config.get('Database', 'db_port')

    # Return a dictionary with the retrieved values
    config_values = {
        'debug_mode': debug_mode,
        'log_level': log_level,
        'db_name': db_name,
        'db_host': db_host,
        'db_port': db_port
    }

    return config_values


if __name__ == "__main__":
    # Call the function to read the configuration file
    config_data = read_config()

    # Print the retrieved values
    print("Debug Mode:", config_data['debug_mode'])
    print("Log Level:", config_data['log_level'])
    print("Database Name:", config_data['db_name'])
    print("Database Host:", config_data['db_host'])
    print("Database Port:", config_data['db_port'])
