# Creation starting dataframes
import numpy as np
import pandas as pd


# --------------------------------------------------------
# -------------- Script Parameters --------------
# --------------------------------------------------------

# File names
path = "./real_data_analysis/results/data"

PATH_TO_PATIENT_DATA = f"{path}/patient_data.csv"
PATH_TO_GENOMICS_DATA = f"{path}/genomics.csv"
PATH_TO_CLINICAL_DATA = f"{path}/clinical.csv"

# column names
PATIENT_ID = "ID"
PATIENT_MEAL_ID = "ID_Meal"
COL_MEAL = "Meal"
COL_VISIT = "Visit"
COL_TIME = "Time"
COL_OUTCOME = "TG"
COL_SEX = "Sex"
COL_AGE = "Age"
COL_BMI = "BMI"


# --------------------------------------------------------
# -------------- Load data --------------
# --------------------------------------------------------

# Load the patient data
df_patient_data = pd.read_csv(PATH_TO_PATIENT_DATA, header=0, sep=";", decimal=",")
df_genomics = pd.read_csv(PATH_TO_GENOMICS_DATA, header=0, sep=";", decimal=",")
df_clinical_data = pd.read_csv(PATH_TO_CLINICAL_DATA, header=0, sep=";", decimal=",", encoding="ISO-8859-1")

# save the genomics column names to file
columns_to_keep = [col for col in df_genomics.columns if col not in [PATIENT_ID, COL_VISIT, COL_TIME]]
genes_names_df = pd.DataFrame({'column_names': columns_to_keep})
genes_names_df.to_csv('genes_names.csv', sep=";", index=False)

# Extract unique IDs from all DFs
patient_data_unique_ids = df_patient_data[PATIENT_ID].unique()
genomics_data_unique_ids = df_genomics[PATIENT_ID].unique()
clinical_data_unique_ids = df_clinical_data[PATIENT_ID].unique()

len_check = len(patient_data_unique_ids) == len(genomics_data_unique_ids) == len(clinical_data_unique_ids)
print(f"Length IDs check: {len_check}")
print(f"len(patient_data_unique_ids): {len(patient_data_unique_ids)}")
print(f"len(genomics_data_unique_ids): {len(genomics_data_unique_ids)}")
print(f"len(clinical_data_unique_ids): {len(clinical_data_unique_ids)}")

# Extract unique mappings from Visit to Meal
meal_mapping_df = df_clinical_data[[PATIENT_ID, COL_VISIT, COL_MEAL]].drop_duplicates()
print(meal_mapping_df)

# First merge genomics with clinical to get the meal value from the visit number
df_genomics = df_genomics.merge(meal_mapping_df, on=[PATIENT_ID, COL_VISIT], how='left')
print(df_genomics.head())
print(f"N unique IDs: {len(df_genomics[PATIENT_ID].unique())}")

# remove nan from this column
df_genomics = df_genomics.dropna(subset = [COL_MEAL])

# merge to align over available IDs
# Perform the left join
df_merged = df_genomics.merge(df_patient_data, on=PATIENT_ID, how='left')
print(df_merged.head())
print(f"N unique IDs: {len(df_merged[PATIENT_ID].unique())}")
# Create new unique ID for patient and meal
df_merged[PATIENT_MEAL_ID] = df_merged.apply(lambda row: f"{row[PATIENT_ID]}_{row[COL_MEAL]}", axis=1)
print(df_merged[PATIENT_MEAL_ID].unique())
print(f"N unique new IDs: {len(df_merged[PATIENT_MEAL_ID].unique())}")
# Change Sex encoding from (1, 2) to (0, 1)
df_merged[COL_SEX] = df_merged[COL_SEX] - 1

# Add PATIENT_MEAL_ID to clinical df
# remove nan from this column
df_clinical_data = df_clinical_data.dropna(subset = [COL_MEAL])

df_clinical_data[PATIENT_MEAL_ID] = df_clinical_data.apply(lambda row: f"{row[PATIENT_ID]}_{row[COL_MEAL]}", axis=1)
print(df_clinical_data[PATIENT_MEAL_ID].unique())
print(f"N unique new IDs: {len(df_clinical_data[PATIENT_MEAL_ID].unique())}")

# Finally keep only rows whose PATIENT_MEAL_ID is in genomics data
df_clinical_filtered = df_clinical_data.loc[df_clinical_data[PATIENT_MEAL_ID].isin(df_merged[PATIENT_MEAL_ID].unique())]
print(f"Shape df_clinical after: {df_clinical_filtered.shape}")
print(f"Missing values for outcome: {df_clinical_filtered[COL_OUTCOME].isna().sum()}")

# Save to csv
df_merged.to_csv("features_data_ready.csv", sep=";", index=False)
df_clinical_filtered.to_csv("clinical_data_ready.csv", sep=";", index=False)
