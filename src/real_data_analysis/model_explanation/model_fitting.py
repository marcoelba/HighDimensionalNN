# Data analysis
import pickle
import os

from src.utils.config_reader import get_config
from src.utils.data_handling.data_loader import CustomData
from src.utils.ensemble_pipeline import EnsemblePipeline
from src.utils.features_preprocessing import Preprocessing

# Script specific modules
# Must be in the same directory where model_fitting.py is run
from full_model import Model


# get config from console input arguments
config_dict = get_config()
os.makedirs(config_dict["script_parameters"]["results_folder"], exist_ok = True)

# Load data
data = CustomData(config_dict, data_dir=config_dict["script_parameters"]["data_folder"])
dict_arrays = data.load_and_process_data(data_dir=config_dict["script_parameters"]["data_folder"])

# preprocessing class
features_preprocessing = Preprocessing(config_dict=config_dict)

# model definition
model_dimension_definition = dict(
    input_dim_genes=data.p_gene,
    input_dim_metab=data.p_metab,
    input_patient_features_dim=data.p_static,
    n_timepoints=data.n_timepoints
)

model_pipeline = EnsemblePipeline(
    Model,
    features_preprocessing,
    config_dict,
    model_dimension_definition
)

# train k-fold pipeline
model_pipeline.train(dict_arrays)

print("\n ---------------------------------------")
print(" ---------- Training finished ----------")
print(" ---------------------------------------")
