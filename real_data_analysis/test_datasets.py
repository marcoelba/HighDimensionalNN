# test datasets
import pandas as pd
import numpy as np


ids = [1, 2, 3, 4, 5, 6, 7, 8]
p = 10
gene_names = [f"gene_{i}" for i in range(p)]

main_data = []
for id_num in ids:
    visits = [1, 2] if id_num in [1, 2, 3] else [1]
    meals = ["A", "B"] if id_num in [1, 2, 3] else ["A"]
    for visit in visits:
        meal = meals[visit - 1]
        for time in [0, 1, 2]:  # Only 2 time points in main df
            a = np.random.randint(1, 10)
            b = np.random.uniform(0, 1)
            c = np.random.uniform(0, 1)
            tg = np.abs(np.random.randn() + 2)
            X = np.random.randn(p)
            row = [id_num, visit, meal, time, a, b, c, tg]
            [row.append(X[j]) for j in range(p)]

            main_data.append(row)

df_main = pd.DataFrame(main_data, columns=[['ID', 'Visit', 'Meal', 'Time', 'Sex', 'Age', 'BMI', 'TG'] + gene_names])

path = "./real_data_analysis/results/data"
df_main.to_csv(f"{path}/features_data_ready.csv", index=False, sep=";")
df_main.to_csv(f"{path}/clinical_data_ready.csv", index=False, sep=";")

genes_names_df = pd.DataFrame({'column_names': gene_names})
genes_names_df.to_csv(f"{path}/genes_names.csv", sep=";", index=False)
