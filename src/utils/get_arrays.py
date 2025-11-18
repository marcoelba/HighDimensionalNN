import os

import numpy as np
import pandas as pd

from src.utils.convert_to_array import convert_to_static_multidim_array, convert_to_longitudinal_multidim_array


def load_and_process_data(config_dict: dict, data_dir: str):

    column_names = config_dict["shared_columns"]
    
    # Load the patient data
    try:
        df_features = pd.read_csv(os.path.join(data_dir, config_dict["file_names"]["path_to_features_data"]), header=0, sep=";")
    except KeyError:
        print("path_to_features_data key NOT found")
    try:
        df_metab_data = pd.read_csv(os.path.join(data_dir, config_dict["file_names"]["path_to_metab_data"]), header=0, sep=";")
    except KeyError:
        print("path_to_features_data key NOT found")
    df_clinical_data = pd.read_csv(os.path.join(data_dir, config_dict["file_names"]["path_to_clinical_data"]), header=0, sep=";")
    df_gene_names = pd.read_csv(os.path.join(data_dir, config_dict["file_names"]["path_to_gene_names"]), header=0, sep=";")
    df_metab_names = pd.read_csv(os.path.join(data_dir, config_dict["file_names"]["path_to_metabolites_names"]), header=0, sep=";")

    # convert genomics data to array
    genes_cols = df_gene_names['column_names'].tolist()
    config_dict["data_arrays"]["genes"] = genes_cols
    metab_cols = df_metab_names['column_names'].tolist()
    config_dict["data_arrays"]["metabolites"] = metab_cols
    # ratio variable should not be standardized
    # config_dict["preprocess"]["metabolites"] = [m for m in metab_cols if not m.startswith("ratio_")]
    
    print("\n-------------------------------")
    print(" Extraction of gene data")
    print(" ------------------------------- ")

    X_gene = convert_to_static_multidim_array(
        df_features,
        baseline_time=0,
        patient_ID_col=column_names["patient_id"],
        visit_col=column_names["col_visit"],
        meal_col=column_names["col_meal"],
        time_index_col=column_names["col_time"],
        cols_to_extract=genes_cols
    )
    print("\ngenes features extracted!")
    print("Shape genes array: ", X_gene.shape)

    X_metab = convert_to_static_multidim_array(
        df_metab_data,
        baseline_time=0,
        patient_ID_col=column_names["patient_id"],
        visit_col=column_names["col_visit"],
        meal_col=column_names["col_meal"],
        time_index_col=column_names["col_time"],
        cols_to_extract=metab_cols
    )
    print("\nMetabs features extracted!")
    print("Shape metab array: ", X_metab.shape)

    # extract static patient features
    print("\n-------------------------------")
    print(" Extraction of patient data")
    print("-------------------------------")

    X_static = convert_to_static_multidim_array(
        df_features,
        baseline_time=0,
        patient_ID_col=column_names["patient_id"],
        visit_col=column_names["col_visit"],
        meal_col=column_names["col_meal"],
        time_index_col=column_names["col_time"],
        cols_to_extract=config_dict["data_arrays"]["static_patient_features"]
    )
    print("\nX_static features extracted!")
    print("Shape patient features array: ", X_static.shape)

    print("\n-------------------------------")
    print(" Extraction of outcome")
    print("-------------------------------")

    y = convert_to_longitudinal_multidim_array(
        df_clinical_data,
        patient_ID_col=column_names["patient_id"],
        visit_col=column_names["col_visit"],
        meal_col=column_names["col_meal"],
        time_index_col=column_names["col_time"],
        cols_to_extract=config_dict["data_arrays"]["y_target"],
        transform=[np.log]
    )
    print("\nOutcome y extracted!")
    print("Shape target array: ", y.shape)

    n_individuals, n_measurements, n_timepoints, _ = y.shape

    print("\n ----------------- Dimensions: ---------------------- ")
    print("n_individuals: ", n_individuals)
    print("n_timepoints: ", n_timepoints - 1)
    print("n_measurements: ", n_measurements)
    print("\n ---------------------------------- ")

    # y0 (y at baseline) is actually an additional feature, because it is measured before any intervention
    y_baseline = y[:, :, 0:1, :]
    # the actual target is then y from t=1
    y_target = y[:, :, 1:, :]

    array_names = list(config_dict["data_arrays"].keys())
    dict_arrays = {
        array_names[0]: X_gene,
        array_names[1]: X_metab,
        array_names[2]: X_static,
        array_names[3]: y_baseline,
        array_names[4]: y_target
    }

    return dict_arrays
