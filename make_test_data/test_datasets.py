# test datasets
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


n = 30
n_bad = 15
p_genes = 10
p_metab = 5
n_time_points = 3

ids = [i + 1 for i in range(n)]

ids_1_meal = [i + 1 for i in range(int(n/2))]
ids_2_meal = np.setdiff1d(ids, ids_1_meal)

ids_bad = np.random.choice(ids, n_bad)
ids_normal = np.setdiff1d(ids, ids_bad)

gene_names = [f"gene_{i}" for i in range(p_genes)]
metab_names = [f"metab_{i}" for i in range(p_metab)]
patient_features_names = ['Sex', 'Age']
clin_features_names = ['BMI', 'TG']

main_data = []
np.random.seed(3546)

for id_num in ids:
    visits = [1, 2] if id_num in ids_2_meal else [1]
    meals = ["A", "B"] if id_num in ids_2_meal else ["A"]
    sex = np.random.randint(0, 2)
    age = np.random.randint(30, 40)
    bmi = np.random.randn() * 0.2 + 1.5
    for visit in visits:
        meal = meals[visit - 1]
        # features measured only at baseline
        tg_baseline = np.exp(np.random.randn() * 0.05)

        if id_num in ids_bad:
            X_genes = np.random.randn(p_genes) * 0.1 + 0.2
            X_metab = np.random.randn(p_metab) * 0.5 + 1.
        else:
            X_genes = np.random.randn(p_genes) * 0.1
            X_metab = np.random.randn(p_metab) * 0.5
        for time in range(n_time_points):
            if time > 0:
                if id_num in ids_bad:
                    tg = tg_baseline + np.random.randn() * 0.1 + 0.2
                else:
                    tg = tg_baseline + np.random.randn() * 0.05
            else:
                tg = tg_baseline
            row = np.concatenate([
                np.array(id_num)[..., None],
                np.array(visit)[..., None],
                np.array(meal)[..., None],
                np.array(time)[..., None],
                np.array(sex)[..., None],
                np.array(age)[..., None],
                np.array(bmi)[..., None],
                np.array(tg)[..., None],
                X_genes,
                X_metab
            ])
            main_data.append(row)

df_main = pd.DataFrame(
    main_data,
    columns=['ID', 'Visit', 'Meal', 'Time'] + patient_features_names + clin_features_names + gene_names + metab_names
)
df_main.shape
columns = df_main.columns
columns_int = [col for col in columns if col != "Meal"]

df_main[columns_int] = df_main[columns_int].apply(pd.to_numeric)

df_group = df_main.groupby(by=["ID", "Visit"], as_index=False).agg(list)

ids_bad
plt.plot(df_group.loc[(df_group["ID"] == 1) & (df_group["Visit"] == 1), "TG"].to_list()[0])
plt.plot(df_group.loc[(df_group["ID"] == 30) & (df_group["Visit"] == 1), "TG"].to_list()[0])
plt.plot(df_group.loc[(df_group["ID"] == 30) & (df_group["Visit"] == 2), "TG"].to_list()[0])
plt.show()

path = "./data"
df_main.to_csv(f"{path}/genomics_data_ready.csv", index=False, sep=";")
df_main.to_csv(f"{path}/metab_data_ready.csv", index=False, sep=";")
df_main.to_csv(f"{path}/clinical_data_ready.csv", index=False, sep=";")

genes_names_df = pd.DataFrame({'column_names': gene_names})
genes_names_df.to_csv(f"{path}/genes_names.csv", sep=";", index=False)

metab_names_df = pd.DataFrame({'column_names': metab_names})
metab_names_df.to_csv(f"{path}/metab_names.csv", sep=";", index=False)
