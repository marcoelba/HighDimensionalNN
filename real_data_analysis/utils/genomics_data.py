# Modify the genomics raw file
import numpy as np
import pandas as pd


PATH_TO_GENOMICS_DATA = "genomics.csv"

# load
df = pd.read_csv(PATH_TO_GENOMICS_DATA, header=0, sep=";")
# Extract the three components using regex
df[['ID', 'Visit', 'Time']] = df['id_visit_time'].str.extract(r'(\d+)v(\d+)t(\d+)')
# drop old one
df.drop('id_visit_time', axis=1, inplace=True)

# save the new csv
df.to_csv("genomics_new.csv", sep=";", index=False)
