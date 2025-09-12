# Data analysis
import numpy as np
import pandas as pd
import torch

import os
os.chdir("./")

from real_data_analysis.convert_to_array import convert_to_multidim_array_efficient

from src.utils import training_wrapper
from src.utils import data_loading_wrappers
from src.utils.model_output_details import count_parameters
from src.utils import plots

from src.vae_attention.full_model import DeltaTimeAttentionVAE


# Load the raw data
df = pd.read_csv('data.csv')

# convert to arrays
convert_to_multidim_array_efficient(
    df: pd.DataFrame,
    patient_ID_col: str,
    meal_col: str,
    time_index_col: str,
    covs_cols: list,
    outcome_col: str
)