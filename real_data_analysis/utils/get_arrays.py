import os

import numpy as np
import pandas as pd

from real_data_analysis.utils.convert_to_array import convert_to_static_multidim_array, convert_to_longitudinal_multidim_array


def load_and_process_data(config_dict: dict, data_dir: str):

    column_names = config_dict["column_names"]
    
    # Load the patient data
    df_features = pd.read_csv(os.path.join(data_dir, config_dict["file_names"]["path_to_features_data"]), header=0, sep=";")
    df_clinical_data = pd.read_csv(os.path.join(data_dir, config_dict["file_names"]["path_to_clinical_data"]), header=0, sep=";")
    df_gene_names = pd.read_csv(os.path.join(data_dir, config_dict["file_names"]["path_to_gene_names"]), header=0, sep=";")

    # convert genomics data to array
    genes_cols = df_gene_names['column_names'].tolist()

    print("\n-------------------------------")
    print(" Extraction of gene data")
    print(" ------------------------------- ")

    X = convert_to_static_multidim_array(
        df_features,
        baseline_time=0,
        patient_ID_col=column_names["patient_id"],
        visit_col=column_names["col_visit"],
        meal_col=column_names["col_meal"],
        time_index_col=column_names["col_time"],
        cols_to_extract=genes_cols
    )
    print("\ngenes features extracted!")
    print("Shape genes array: ", X.shape)

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
        cols_to_extract=column_names["patient_cols"]
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
        cols_to_extract=[column_names["col_outcome"]],
        transform=[np.log]
    )
    print("\nOutcome y extracted!")
    print("Shape target array: ", y.shape)

    n_individuals, n_measurements, n_timepoints, _ = y.shape
    p = X.shape[-1]
    p_static = X_static.shape[-1]

    print("\n ---------------------------------- ")
    print("Dimensions:")
    print("n_individuals: ", n_individuals)
    print("n_timepoints: ", n_timepoints)
    print("n_measurements: ", n_measurements)
    print("p: ", p)
    print("p_static: ", p_static)
    print("\n ---------------------------------- ")

    # y0 (y at baseline) is actually an additional feature, because it is measured before any intervention
    y_baseline = y[:, :, 0:1, :]
    # the actual target is then y from t=1
    y_target = y[:, :, 1:, :]

    # Add preproc info
    features_to_preprocess = config_dict["preprocess"]

    features_to_preprocess = dict()
    features_to_preprocess[config_dict["array_names"]['genes']] = {k: v for v, k in enumerate(genes_cols)}
    features_to_preprocess[config_dict["array_names"]['static_patient_features']] = dict(COL_AGE=1, COL_BMI=2)
    features_to_preprocess[config_dict["array_names"]["y_baseline"]] = dict(COL_OUTCOME=0)
    features_to_preprocess[config_dict["array_names"]["y_target"]] = dict(COL_OUTCOME=0)

    n_timepoints = n_timepoints - 1
    print("n_timepoints withOUT baseline: ", n_timepoints)

    dict_arrays = dict(
        genes=X,
        static_patient_features=X_static,
        y_baseline=y_baseline,
        y_target=y_target
    )

    return dict_arrays, features_to_preprocess
