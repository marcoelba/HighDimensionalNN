import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


genes_latent = pd.read_csv("/home/marco_ocbe/Documents/UiO_Postdoc/Code/git_repositories/HighDimNN/results/genes_latent_shap.csv", sep=";", index_col=0)
metab_latent = pd.read_csv("/home/marco_ocbe/Documents/UiO_Postdoc/Code/git_repositories/HighDimNN/results/metab_latent_shap.csv", sep=";", index_col=0)


def create_shap_network(shap_matrix, feature_names, latent_names=None, threshold=0.1):
    """
    Create bipartite network from SHAP matrix
    
    shap_matrix: [p_features, K_dims] - mean absolute SHAP values
    threshold: minimum SHAP value to include an edge
    """
    if latent_names is None:
        latent_names = [f'Latent_{i}' for i in range(shap_matrix.shape[1])]
    
    # Create bipartite graph
    G = nx.Graph()
    
    # Add nodes with attributes
    for i, feature in enumerate(feature_names):
        G.add_node(feature, bipartite=0, node_type='feature')  # bipartite=0 for features
    
    for j, latent in enumerate(latent_names):
        G.add_node(latent, bipartite=1, node_type='latent')   # bipartite=1 for latent dims
    
    # Add edges based on SHAP importance
    for i, feature in enumerate(feature_names):
        for j, latent in enumerate(latent_names):
            shap_value = shap_matrix[i, j]
            if shap_value >= threshold:
                G.add_edge(feature, latent, weight=shap_value, 
                          importance=shap_value)
    
    return G

# Create the network
genes_names = list(genes_latent.index)
genes_latent.to_numpy().mean()
genes_latent.to_numpy().std()

G_genes = create_shap_network(genes_latent.to_numpy(), genes_names, threshold=0.008)

print(f"Network stats:")
print(f"  Features: {len([n for n in G_genes.nodes() if G_genes.nodes[n]['node_type'] == 'feature'])}")
print(f"  Latent dims: {len([n for n in G_genes.nodes() if G_genes.nodes[n]['node_type'] == 'latent'])}")
print(f"  Edges: {G_genes.number_of_edges()}")

def plot_network_simple(G):
    """
    Simplest version - just pass the graph
    """
    plt.figure(figsize=(14, 10))
    
    # Get nodes
    features = [n for n in G.nodes() if G.nodes[n]['node_type'] == 'feature']
    latents = [n for n in G.nodes() if G.nodes[n]['node_type'] == 'latent']
    
    # Create layout
    pos = {}
    for i, f in enumerate(features):
        pos[f] = (0, i)
    for i, l in enumerate(latents):
        pos[l] = (1, i * len(features) / len(latents))
    
    # Draw edges with proportional widths
    for u, v in G.edges():
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], 
                              width=G[u][v]['weight'] * 20,
                              alpha=0.6, edge_color='black')
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, nodelist=features, node_color='blue', node_size=400)
    nx.draw_networkx_nodes(G, pos, nodelist=latents, node_color='red', node_size=600)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.title("Feature → Latent Dimension Connections")
    plt.axis('off')
    plt.show()

# Simplest usage
plot_network_simple(G_genes)
