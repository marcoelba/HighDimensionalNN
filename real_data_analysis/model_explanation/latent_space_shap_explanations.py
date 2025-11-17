# Latent space SHAP explanations
import pickle
import os
import copy

import shap
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from real_data_analysis.convert_to_array import convert_to_static_multidim_array, convert_to_longitudinal_multidim_array
from real_data_analysis.features_preprocessing import preprocess, preprocess_transform
from src.utils import data_loading_wrappers
from real_data_analysis.config_reader import read_config
from real_data_analysis.get_arrays import load_and_process_data


# Read config
PATH_MODELS = "./res_train_v2"
PATH_PLOTS = "plots_vae_latent"
os.makedirs(PATH_PLOTS, exist_ok = True)

config_dict = read_config("./model_use_vae_z/config.ini")
DEVICE = torch.device(config_dict["training_parameters"]["device"])
N_FOLDS = config_dict["training_parameters"]["n_folds"]
FEATURES_KEYS = list(config_dict["preprocess"].keys())[:-1]

# --------------------------------------------------------
# -------------------- Load data -------------------------
# --------------------------------------------------------
dict_arrays, features_to_preprocess = load_and_process_data(config_dict, data_dir="./")
n_individuals = dict_arrays["genes"].shape[0]
p = dict_arrays["genes"].shape[2]
p_static = dict_arrays["static_patient_features"].shape[2]
n_timepoints = dict_arrays["y_target"].shape[2]

# Load pickle files
with open(f"{PATH_MODELS}/all_scalers", "rb") as fp:   # Pickling scalers
    all_scalers = pickle.load(fp)

genes_names = pd.read_csv("genes_names.csv", header=0, sep=";")
genes_names = genes_names["column_names"].to_numpy()

# Load torch models
all_models = []
for fold in range(N_FOLDS):
    print(f"Loading model fold {fold+1} of {N_FOLDS}")

    PATH = f"{PATH_MODELS}/model_{fold}"
    model = DeltaTimeAttentionVAE(
        input_dim=p,
        patient_features_dim=p_static,
        n_timepoints=n_timepoints,
        vae_latent_dim=config_dict["model_params"]["latent_dim"],
        vae_input_to_latent_dim=64,
        max_len_position_enc=10,
        transformer_input_dim=config_dict["model_params"]["transformer_input_dim"],
        transformer_dim_feedforward=config_dict["model_params"]["transformer_dim_feedforward"],
        nheads=config_dict["model_params"]["n_heads"],
        dropout=0.1,
        dropout_attention=0.1,
        prediction_weight=1.0
    ).to(DEVICE)
    model.load_state_dict(torch.load(PATH))
    all_models.append(model)


# Define a Torch ensemble model that takes in input a list of models
class EnsembleModelVae(torch.nn.Module):
    def __init__(self, model_list, latent_dim_to_explain):
        super(EnsembleModelVae, self).__init__()
        self.models = torch.nn.ModuleList(model_list)
        self.latent_dim_to_explain = latent_dim_to_explain

    def forward(self, x):
        all_outputs = []
        for model in self.models:
            model.eval()
            output = model.vae.encoder(x)[:, [self.latent_dim_to_explain]]
            all_outputs.append(output)
        return torch.stack(all_outputs).mean(dim=0)

# pre-process the input data with all folds scalers at once
def prepare_data_for_shap(dict_shap, subsample=False, n_background=100):
    tensor_input_per_fold = []

    for fold in range(N_FOLDS):
        # apply feature preprocessing
        dict_arrays_preproc = preprocess_transform(
            copy.deepcopy(dict_shap), all_scalers[fold], features_to_preprocess
        )
        if dict_arrays_preproc["y_baseline"].shape[-1] == 1:
            dict_arrays_preproc["y_baseline"] = dict_arrays_preproc["y_baseline"][..., 0]
        
        if subsample:
            index = np.random.choice(next(iter(dict_shap.values())).shape[0], n_background, replace=False)
            tensor_data = [
                torch.FloatTensor(array[index]).to(DEVICE) for key, array in dict_arrays_preproc.items()
            ]
        else:
            tensor_data = [
                torch.FloatTensor(array).to(DEVICE) for key, array in dict_arrays_preproc.items()
            ]
        
        # make longitudinal
        tensor_dataset = data_loading_wrappers.CustomDataset(
            *tensor_data,
            reshape=True,
            remove_missing=True,
            feature_dimensions=-1,
            device=DEVICE
        )
        tensor_input = tensor_dataset.arrays
        # append
        tensor_input_per_fold.append(tensor_input)
    
    combined_tensors_per_fold = []
    for feature in range(len(FEATURES_KEYS)):
        tensor_feature = []
        for fold in range(N_FOLDS):
            tensor_feature.append(tensor_input_per_fold[fold][feature])
        combined_tensors_per_fold.append(torch.cat(tensor_feature, dim=0))
    
    return combined_tensors_per_fold


# -------------------------------------------------------------------------
# ---------------------- Run SHAP explanation -----------------------------
# -------------------------------------------------------------------------
print("---------------- Running SHAP on VAE Latent space ---------------")
dict_shap = {key: array for key, array in dict_arrays.items() if key in FEATURES_KEYS}
background_data = prepare_data_for_shap(dict_shap, subsample=True, n_background=25)[0]
print("Shape background_data for SHAP: ", background_data.shape)
explain_data = prepare_data_for_shap(dict_shap, subsample=False)[0]
print("Shape explain daata for SHAP: ", explain_data.shape)

vae_latent_shap_values = []
for latent_dim_to_explain in range(config_dict["model_params"]["latent_dim"]):
    ensemble_model = EnsembleModelVae(all_models, latent_dim_to_explain=latent_dim_to_explain)
    explainer = shap.GradientExplainer(ensemble_model, background_data)
    shap_values = explainer.shap_values(explain_data)[..., -1]
    vae_latent_shap_values.append(shap_values)

    # Plot
    fig = plt.figure()
    shap.summary_plot(
        shap_values,
        features=explain_data,
        feature_names=genes_names,
        show = False
    )
    fig.savefig(f"{PATH_PLOTS}/shap_latent_dim_{latent_dim_to_explain}.pdf", format="pdf")

# save shap values ot pickle
with open("vae_latent_shap_values", "wb") as fp:
    pickle.dump(vae_latent_shap_values, fp)

# Check which genes share the same latent dimensions
with open(f"{PATH_MODELS}/vae_latent_shap_values", "rb") as fp:
    shap_values = pickle.load(fp)
len(shap_values)

k_top = 5
top_for_latent_dim = []
for latent_dim_to_explain in range(len(shap_values)):
    sl = shap_values[latent_dim_to_explain]
    sl_mean = np.abs(sl).mean(axis=0)
    sl_sort = np.sort(sl_mean)[::-1]
    # plt.plot(sl_sort)
    # plt.show()
    sl_argsort = np.argsort(sl_mean)[::-1]
    top_ten = genes_names[sl_argsort][0:k_top]
    top_for_latent_dim.append(top_ten)

unique_genes = np.unique(np.array(top_for_latent_dim).flatten())
unique_genes.shape
latent_df = pd.DataFrame(np.zeros([len(shap_values), unique_genes.shape[0]]), columns=unique_genes)

for latent_dim_to_explain in range(len(shap_values)):
    for gene in top_for_latent_dim[latent_dim_to_explain]:
        latent_df[gene][latent_dim_to_explain] = 1

latent_df.sum(axis=0)

co_occurrence = latent_df.T.dot(latent_df)  # Features x Features matrix
np.fill_diagonal(co_occurrence.values, 0)  # Remove self-loops

# network plot
import networkx as nx

# Create graph
G = nx.Graph()

# Add nodes (features)
for feature in latent_df.columns:
    G.add_node(feature)

# Add edges with weights based on co-occurrence
for i, feature1 in enumerate(latent_df.columns):
    for j, feature2 in enumerate(latent_df.columns):
        if i < j and co_occurrence.loc[feature1, feature2] > 0:
            G.add_edge(feature1, feature2, weight=co_occurrence.loc[feature1, feature2])

# Visualize
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G)

# Draw with edge thickness proportional to co-occurrence
edges = G.edges()
weights = np.array([G[u][v]['weight'] for u,v in edges])
null_weight = 0.05
weights[weights == 1] = null_weight
edge_colors = ["red" if w > null_weight else "black" for w in weights]

nx.draw_networkx_nodes(G, pos, node_size=200, node_color='lightblue', alpha=0.8)
nx.draw_networkx_edges(G, pos, width=weights, alpha=0.9, edge_color=edge_colors)
nx.draw_networkx_labels(G, pos, font_size=10)

plt.title("Feature Co-occurrence Network")
plt.axis('off')
plt.show()



# Create bipartite graph
B = nx.Graph()

# Add group nodes and feature nodes
B.add_nodes_from(latent_df.index, bipartite=0)  # Groups
B.add_nodes_from(latent_df.columns, bipartite=1)  # Features

# Add edges where feature appears in group
for group in latent_df.index:
    for feature in latent_df.columns:
        if latent_df.loc[group, feature] == 1:
            B.add_edge(group, feature)

# Separate node types
group_nodes = [n for n in B.nodes() if B.nodes[n]['bipartite'] == 0]
feature_nodes = [n for n in B.nodes() if B.nodes[n]['bipartite'] == 1]

plt.figure(figsize=(14, 10))
pos = nx.bipartite_layout(B, group_nodes)

nx.draw_networkx_nodes(B, pos, nodelist=group_nodes, 
                      node_color='red', node_size=300, alpha=0.7)
nx.draw_networkx_nodes(B, pos, nodelist=feature_nodes, 
                      node_color='blue', node_size=500, alpha=0.7)
nx.draw_networkx_edges(B, pos, alpha=0.3)
nx.draw_networkx_labels(B, pos, font_size=8)

plt.title("Bipartite Network: Groups and Features")
plt.axis('off')
plt.show()
