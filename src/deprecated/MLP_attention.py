# MLP with attention layer
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import StandardScaler
from prettytable import PrettyTable

import matplotlib.pyplot as plt
import shap
from scipy.stats import pearsonr

import os
os.chdir("./src")

from vae_regression.data_generation import data_generation
from model_utils import utils


class MLPWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super().__init__()
        self.input_dim = input_dim
        
        # Feature-wise attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),  # Projects to per-gene scores
            nn.Softmax(dim=1)             # Normalizes to [0,1] per sample
        )

        # MLP backbone
        self.mlp = nn.Sequential(
            # nn.ReLU(),
            nn.Linear(input_dim, hidden_dim),
            # nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):

        # Get attention weights (n_samples, n_genes)
        attn_weights = self.attention(x)
        
        # Apply attention: x * attn_weights (element-wise)
        weighted_x = x * attn_weights
        
        # Pass through MLP
        output = self.mlp(weighted_x)
        
        return output, attn_weights


# generate data assuming an underlying latent space of dimension k
k = 5
n_train = 1000
n_test = 200
n_val = 200
n = n_train + n_test + n_val
p = 40

# custom W
W = np.random.normal(size=(k, p))
first_half = range(0, int(p/2))
second_half = range(int(p/2), p)
# block structure
W[0, first_half] = 0.0
W[1, first_half] = 0.0
W[3, second_half] = 0.0
W[4, second_half] = 0.0
# first 10 features do NOT have any effect
W[:, 0:10] = 0

plt.plot(W[0, :], linestyle="", marker="o", label="k1")
plt.plot(W[3, :], linestyle="", marker="o", label="k3")
plt.show()

beta = np.array([-1, 1, -1, 1, -1])

y, X, Z, beta = data_generation(n, k, p, noise_scale = 0.5, beta=beta, W=W)


# get tensors
X_tensor = torch.FloatTensor(X).to(torch.device("cpu"))
Z_tensor = torch.FloatTensor(Z).to(torch.device("cpu"))
y_tensor = torch.FloatTensor(y).to(torch.device("cpu"))

# split data
data_split = utils.DataSplit(X.shape[0], test_size=n_test, val_size=n_val)
print("train: ", len(data_split.train_index), 
    "val: ", len(data_split.val_index),
    "test: ", len(data_split.test_index)
)


data_train = data_split.get_train(X, Z, y)
data_test = data_split.get_test(X, Z, y)
data_val = data_split.get_val(X, Z, y)

tensor_data_train = data_split.get_train(X_tensor, y_tensor)
tensor_data_test = data_split.get_test(X_tensor, y_tensor)
tensor_data_val = data_split.get_val(X_tensor, y_tensor)

# make tensor data loaders
train_dataloader = utils.make_data_loader(*tensor_data_train, batch_size=32)
test_dataloader = utils.make_data_loader(*tensor_data_test, batch_size=32)
val_dataloader = utils.make_data_loader(*tensor_data_val, batch_size=32)


class Training:
    def __init__(self, train_dataloader, val_dataloader=None):

        self.losses = dict()
        
        self.validation = (val_dataloader is not None)
        self.train_dataloader = train_dataloader
        self.len_train = len(train_dataloader.dataset)
        self.losses["train"] = []

        if self.validation:
            self.val_dataloader = val_dataloader
            self.len_val = len(val_dataloader.dataset)
            self.losses["val"] = []
        else:
            self.losses["val"] = None

        self.best_val_loss = float('inf')
        self.best_model = None

    def training_loop(self, model, optimizer, loss_criterion, num_epochs, lambda_l1=None):

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0

            for batch_x, batch_y in self.train_dataloader:
                optimizer.zero_grad()
                
                pred_y, _ = model(batch_x)
                loss = loss_criterion(pred_y, batch_y)
                l1_penalty = lambda_l1 * torch.sum(torch.abs(model.attention[0].weight))
                tot_loss = loss + l1_penalty
                tot_loss.backward()

                train_loss += loss.item()
                optimizer.step()
            
            self.losses["train"].append(train_loss / self.len_train)
            print(f'Epoch {epoch+1}, Loss: {train_loss / self.len_train:.4f}')
            
            if self.validation:
                val_loss = 0
                with torch.no_grad():
                    for batch_x, batch_y in self.val_dataloader:
                        
                        pred_y, _ = model(batch_x)
                        loss = loss_criterion(pred_y, batch_y)
                        val_loss += loss.item()
                self.losses["val"].append(val_loss / self.len_val)
                print(f'Epoch {epoch+1}, Validation Loss: {val_loss / self.len_val:.4f}')
            
                # save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_model = copy.deepcopy(model)


# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MLPWithAttention(input_dim=p, hidden_dim=5).to(device)
loss_criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
num_epochs = 200

trainer = Training(train_dataloader, val_dataloader)
trainer.losses.keys()

trainer.training_loop(
    model,
    optimizer,
    loss_criterion,
    num_epochs,
    lambda_l1=0.01
)

plt.plot(trainer.losses["train"], label="train")
plt.plot(trainer.losses["val"], label="val")
plt.vlines(np.argmin(trainer.losses["val"]), 0, max(trainer.losses["val"]), color="red")
plt.hlines(np.min(trainer.losses["val"]), 0, num_epochs, color="red", linestyles="--")
plt.legend()
plt.show()

model.load_state_dict(trainer.best_model.state_dict())

# test predictions
model.eval()
with torch.no_grad():
    y_test_hat, attn_weights = model(tensor_data_test[0])

pearsonr(data_test[2], y_test_hat)[0]
np.sqrt(np.mean((y_test_hat.numpy() - data_test[2])**2))


# Get mean attention weights across all test samples
attn_weights.shape  # n x p
attn_weights = attn_weights.numpy()
mean_attn = np.mean(attn_weights, axis=0)  # Shape: (n_genes,)
mean_attn.shape

plt.plot(mean_attn, linestyle="", marker="o")
plt.show()

# Visualize attention weights (example for the first test sample)
plt.figure(figsize=(12, 4))
plt.bar(range(p), attn_weights[1])
plt.xlabel("Gene Index")
plt.ylabel("Attention Weight")
plt.title("Attention Weights for First Test Sample")
plt.show()

input_to_k = model.mlp[0].weight.detach().numpy()
input_to_k.shape

k_comp = 0
plt.plot(input_to_k[k_comp, :], linestyle="", marker="o")
plt.show()

# attention
attention_w = model.attention[0].weight.detach().numpy()
attention_w.shape

att_comp = 2
plt.plot(attention_w[att_comp, :], linestyle="", marker="o")
plt.show()

# final layer
k_to_y = model.mlp[1].weight.detach().numpy()
k_to_y.shape
k_to_y


# SHAP
model.eval()

# Define a prediction function for the outcome y
def predict(x):
    x = torch.FloatTensor(x)
    with torch.no_grad():
        preds, _ = model(x)
    return preds.numpy().flatten()

# Create KernelExplainer
explainer = shap.KernelExplainer(predict, data_train[0])  # Using 100 samples as background
# Deep explainer for NN
# model.eval()
# explainer = shap.DeepExplainer((model, model.fc1), X_train_tensor)

samples_to_explain = data_test[0]
samples_to_explain.shape

shap_values = explainer.shap_values(samples_to_explain)
shap_values.shape

# Plot feature importance
feature_names = [f'Feature {i}' for i in range(p)]

shap.summary_plot(shap_values, samples_to_explain, show = True)
shap.summary_plot(shap_values, samples_to_explain, plot_type="violin", show = True)

shap.summary_plot(shap_values, samples_to_explain, 
    show = True, feature_names=feature_names[0:10]
)

plt.plot(shap_values[0, :], linestyle="", marker="o")
plt.show()
