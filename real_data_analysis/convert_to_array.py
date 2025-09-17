# Convert long format data to multidimensional arrays
import numpy as np
import pandas as pd

def convert_to_static_multidim_array(
    df: pd.DataFrame,
    baseline_time: int,
    patient_ID_col: str,
    visit_col: str,
    meal_col: str,
    time_index_col: str,
    cols_to_extract: list
):
    """
    Convert a longitudinal DataFrame to multi-dimensional arrays for analysis.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing longitudinal data
    baseline_time: int
        Value of baseline time point
    patient_ID_col : str
        Column name for patient/subject IDs
    meal_col : str
        Column name for meal identifiers
    visit_col : str
        Column name for visit identifiers
    time_index_col : str
        Column name for time points
    cols_to_extract : list
        List of column names for covariates/features
        
    Returns
    -------
    X : ndarray
        Array of shape (n_subjects, n_meals, n_covariates)
        Contains NaN for missing values
    """
    
    # Filter to only include Time = 1
    df_time1 = df[df[time_index_col] == baseline_time].copy()

    # Get unique IDs and maximum number of visits
    unique_ids = sorted(df[patient_ID_col].unique())
    unique_meals = sorted(df[meal_col].dropna().unique())
    max_visits = len(unique_meals)
    n_features = len(cols_to_extract)

    # unique meal to number mapping
    meal_to_idx = {meal: idx for idx, meal in enumerate(unique_meals)}

    # Initialize the array with NaN values
    result_array = np.full((len(unique_ids), max_visits, n_features), np.nan)

    # Fill the array with actual values
    for idx, id_num in enumerate(unique_ids):
        # Get all rows for this ID with Time = 1
        id_data = df_time1[df_time1[patient_ID_col] == id_num]
        
        for _, row in id_data.iterrows():
            meal = row[meal_col]
            for cov_idx, cov in enumerate(cols_to_extract):
                result_array[idx, meal_to_idx[meal], cov_idx] = row[cov]

    print("Original DataFrame:")
    print(df.head())
    print(f"\nResult array shape: {result_array.shape}")
    print("Result array:")
    print(result_array[0])
    print(result_array[1])

    return result_array


def convert_to_longitudinal_multidim_array(
    df: pd.DataFrame,
    patient_ID_col: str,
    visit_col: str,
    meal_col: str,
    time_index_col: str,
    cols_to_extract: list
):
    """
    Convert a longitudinal DataFrame to multi-dimensional arrays for analysis,
    retaining all time points.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing longitudinal data
    patient_ID_col : str
        Column name for patient/subject IDs
    meal_col : str
        Column name for meal identifiers
    visit_col : str
        Column name for visit number identifiers
    time_index_col : str
        Column name for time points
    cols_to_extract : list
        List of column names for covariates/features
        
    Returns
    -------
    X : ndarray
        Array of shape (n_subjects, n_meals, n_timepoints, n_covariates)
        Contains NaN for missing values
    """
    
    # Get unique values for each dimension
    unique_ids = sorted(df[patient_ID_col].unique())
    unique_meals = sorted(df[meal_col].unique())
    unique_times = sorted(df[time_index_col].unique())
    
    n_subjects = len(unique_ids)
    n_meals = len(unique_meals)
    n_timepoints = len(unique_times)
    n_features = len(cols_to_extract)

    # Initialize the array with NaN values
    result_array = np.full((n_subjects, n_meals, n_timepoints, n_features), np.nan)

    # Create mapping dictionaries for faster lookup
    id_to_idx = {id_val: idx for idx, id_val in enumerate(unique_ids)}
    meal_to_idx = {meal_val: idx for idx, meal_val in enumerate(unique_meals)}
    time_to_idx = {time_val: idx for idx, time_val in enumerate(unique_times)}

    # Fill the array with actual values
    for _, row in df.iterrows():
        id_idx = id_to_idx[row[patient_ID_col]]
        meal_idx = meal_to_idx[row[meal_col]]
        time_idx = time_to_idx[row[time_index_col]]
        
        for cov_idx, cov in enumerate(cols_to_extract):
            result_array[id_idx, meal_idx, time_idx, cov_idx] = row[cov]

    print("Original DataFrame:")
    print(df.head())
    print(f"\nResult array shape: {result_array.shape}")
    print(f"Dimensions: {n_subjects} subjects × {n_meals} meals × {n_timepoints} timepoints × {n_features} features")
    
    # Print a sample of the array
    print("\nSample of result array (first 2 subjects, first meal, all timepoints):")
    for i in range(min(2, n_subjects)):
        print(f"Subject {i}:")
        print(result_array[i, 0, :, :])
        print()

    if len(cols_to_extract) == 1:
        result_array = result_array[..., 0]
    
    return result_array
