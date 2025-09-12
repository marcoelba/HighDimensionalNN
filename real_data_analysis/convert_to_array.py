# Convert long format data to multidimensional arrays
import numpy as np
import pandas as pd

def convert_to_multidim_array_efficient(
    df: pd.DataFrame,
    patient_ID_col: str,
    meal_col: str,
    time_index_col: str,
    covs_cols: list,
    outcome_col: str
):
    """
    Convert a longitudinal DataFrame to multi-dimensional arrays for analysis.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing longitudinal data
    patient_ID_col : str
        Column name for patient/subject IDs
    meal_col : str
        Column name for meal/visit identifiers
    time_index_col : str
        Column name for time points
    covs_cols : list
        List of column names for covariates/features
    outcome_col : str
        Column name for the outcome/target variable
    
    Returns
    -------
    y : ndarray
        Outcome array of shape (n_subjects, n_timepoints, n_meals)
        Contains NaN for missing values
    X : ndarray
        Covariate array of shape (n_subjects, n_covariates, n_meals)
        Contains NaN for missing values
    """
        
    print(f"Dataframe size: {df.shape}")
    
    all_IDs = sorted(df[patient_ID_col].unique())
    n = len(all_IDs)
    print(f"N. subjects: {n}")
    
    visit_meals = sorted(df[meal_col].unique())
    M = len(visit_meals)
    print(f"N. meals: {M}")
    
    time_points = sorted(df[time_index_col].unique())
    T = len(time_points)
    print(f"N. time points: {T}")
    
    p = len(covs_cols)
    print(f"N. covariates: {p}")
    
    # Initialize arrays with NaN
    y = np.full((n, T, M), np.nan)
    X = np.full((n, p, M), np.nan)
    
    # Create mapping dictionaries
    id_to_idx = {id_val: idx for idx, id_val in enumerate(all_IDs)}
    meal_to_idx = {meal_val: idx for idx, meal_val in enumerate(visit_meals)}
    time_to_idx = {time_val: idx for idx, time_val in enumerate(time_points)}
    
    # Process X (covariates) - one value per subject per meal
    # Group by patient and meal, take first occurrence for covariates
    x_groups = df.groupby([patient_ID_col, meal_col])[covs_cols].first()
    
    for (id_val, meal_val), row in x_groups.iterrows():
        if id_val in id_to_idx and meal_val in meal_to_idx:
            id_idx = id_to_idx[id_val]
            meal_idx = meal_to_idx[meal_val]
            for ii, cov in enumerate(covs_cols):
                X[id_idx, ii, meal_idx] = row[cov]
    
    # Process y (outcomes) - can vary by time
    # Pivot table for easier access
    y_pivot = df.pivot_table(
        values=outcome_col,
        index=[patient_ID_col, meal_col],
        columns=time_index_col,
        aggfunc='first'  # take first value if multiple entries
    )
    
    for (id_val, meal_val), row in y_pivot.iterrows():
        if id_val in id_to_idx and meal_val in meal_to_idx:
            id_idx = id_to_idx[id_val]
            meal_idx = meal_to_idx[meal_val]
            for time_val, y_val in row.items():
                if pd.notna(y_val) and time_val in time_to_idx:
                    time_idx = time_to_idx[time_val]
                    y[id_idx, time_idx, meal_idx] = y_val
    
    return y, X
