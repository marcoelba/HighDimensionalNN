import os
from collections import OrderedDict

import numpy as np
import pandas as pd

from src.utils.data_handling.convert_to_array import convert_to_static_multidim_array, convert_to_longitudinal_multidim_array


class CustomData:
    def __init__(self, config_dict, data_dir):
        self.config_dict = config_dict

        genes_names = pd.read_csv(os.path.join(data_dir, "genes_names.csv"), header=0, sep=";")
        metab_names = pd.read_csv(os.path.join(data_dir, "metab_names.csv"), header=0, sep=";")
        self.array_names = list(config_dict['data_arrays'].keys())
        self.features_names = OrderedDict()
        self.features_names[self.array_names[0]] = genes_names["column_names"].to_numpy()
        self.features_names[self.array_names[1]] = metab_names["column_names"].to_numpy()

        self.n_individuals = None
        self.n_measurements = None
        self.n_timepoints = None

        self.meal_idx_mapping = None

    def load_and_process_data(self, data_dir: str):

        column_names = self.config_dict["shared_columns"]
        
        # Load the patient data
        df_features = pd.read_csv(os.path.join(data_dir, self.config_dict["file_names"]["path_to_features_data"]), header=0, sep=";")
        df_metab_data = pd.read_csv(os.path.join(data_dir, self.config_dict["file_names"]["path_to_metab_data"]), header=0, sep=";")
        df_clinical_data = pd.read_csv(os.path.join(data_dir, self.config_dict["file_names"]["path_to_clinical_data"]), header=0, sep=";")

        # self.config_dict["data_arrays"]["genes"] = genes_cols
        # self.config_dict["data_arrays"]["metabolites"] = metab_cols
        
        print("\n-------------------------------")
        print(" Extraction of gene data")

        X_gene, extracted_cols = convert_to_static_multidim_array(
            df_features,
            baseline_time=0,
            patient_ID_col=column_names["patient_id"],
            visit_col=column_names["col_visit"],
            meal_col=column_names["col_meal"],
            time_index_col=column_names["col_time"],
            cols_to_extract=self.features_names[self.array_names[0]]
        )
        print("\ngenes features extracted!")
        print("Shape genes array: ", X_gene.shape)

        if X_gene.shape[-1] != len(self.features_names[self.array_names[0]]):
            raise ValueError("Genes dimension != genes names length")
        self.p_gene = X_gene.shape[-1]

        X_metab, extracted_cols = convert_to_static_multidim_array(
            df_metab_data,
            baseline_time=0,
            patient_ID_col=column_names["patient_id"],
            visit_col=column_names["col_visit"],
            meal_col=column_names["col_meal"],
            time_index_col=column_names["col_time"],
            cols_to_extract=self.features_names[self.array_names[1]]
        )
        print("\nMetabs features extracted!")
        print("Shape metab array: ", X_metab.shape)

        if X_metab.shape[-1] != len(self.features_names[self.array_names[1]]):
            raise ValueError("Metab dimension != metabolites names length")
        self.p_metab = X_metab.shape[-1]

        # extract static patient features
        print(" Extraction of patient data")
        print("-------------------------------")

        X_static, extracted_cols = convert_to_static_multidim_array(
            df_features,
            baseline_time=0,
            patient_ID_col=column_names["patient_id"],
            visit_col=column_names["col_visit"],
            meal_col=column_names["col_meal"],
            time_index_col=column_names["col_time"],
            cols_to_extract=self.config_dict["data_arrays"][self.array_names[2]]
        )
        print("\nX_static features extracted!")
        print("Shape patient features array: ", X_static.shape)
        self.p_static = X_static.shape[-1]
        # update feature_names
        self.features_names[self.array_names[2]] = extracted_cols
        _ = self.get_meal_mapping(df_features, column_names["col_meal"])

        print("\n-------------------------------")
        print(" Extraction of outcome")

        y = convert_to_longitudinal_multidim_array(
            df_clinical_data,
            patient_ID_col=column_names["patient_id"],
            visit_col=column_names["col_visit"],
            meal_col=column_names["col_meal"],
            time_index_col=column_names["col_time"],
            cols_to_extract=self.config_dict["data_arrays"][self.array_names[3]],
            transform=[np.log]
        )
        print("\nOutcome y extracted!")
        print("Shape target array: ", y.shape)

        self.n_individuals, self.n_measurements, self.n_timepoints, _ = y.shape
        self.n_timepoints = self.n_timepoints - 1

        print("\n ----------------- Dimensions: ---------------------- ")
        print("n_individuals: ", self.n_individuals)
        print("n_timepoints: ", self.n_timepoints)
        print("n_measurements: ", self.n_measurements)
        print("\n ---------------------------------- ")

        # y0 (y at baseline) is actually an additional feature, because it is measured before any intervention
        y_baseline = y[:, :, 0:1, :]
        self.features_names[self.array_names[3]] = "TG baseline"
        # the actual target is then y from t=1
        y_target = y[:, :, 1:, :]
        self.features_names[self.array_names[4]] = "TG"

        dict_arrays = {
            self.array_names[0]: X_gene,
            self.array_names[1]: X_metab,
            self.array_names[2]: X_static,
            self.array_names[3]: y_baseline,
            self.array_names[4]: y_target
        }

        return dict_arrays

    def get_indeces(self, dict_arrays):
        where_not_na = [sum_not_nan(item) > 0 for key, item in dict_arrays.items()]
        where_all = np.prod(np.array(where_not_na), axis=0)
        print("\n Total not NAs: ", where_all.sum())
        
        return where_all
    
    def get_meal_mapping(self, df, meal_col):
        unique_meals = sorted(df[meal_col].dropna().unique())
        # unique meal to number mapping
        meal_to_idx = {meal: idx for idx, meal in enumerate(unique_meals)}
        self.meal_idx_mapping = meal_to_idx
        
        return meal_to_idx


def sum_not_nan(x):
    x_shape = x.shape
    return (~np.isnan(x)).sum(axis=-1) if x.shape[-1] > 1 else (~np.isnan(x)).sum(axis=-1).sum(axis=-1)
