# Data analysis
import pickle
import os

from src.utils.config_reader import get_config
from src.utils.data_handling.data_loader import CustomData


# get config from console input arguments
config_dict = get_config()
os.makedirs(config_dict["script_parameters"]["results_folder"], exist_ok = True)

# Load data
data = CustomData(config_dict, data_dir=config_dict["script_parameters"]["data_folder"])
dict_arrays = data.load_and_process_data(data_dir=config_dict["script_parameters"]["data_folder"])
patient_indeces = data.get_indeces(dict_arrays)

with open(f"{config_dict["script_parameters"]["results_folder"]}/patient_indeces", "wb") as fp:
    pickle.dump(patient_indeces, fp)
