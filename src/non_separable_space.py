# separable space
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.parametrizations import orthogonal

import numpy as np
from scipy.linalg import toeplitz
from scipy.stats import pearsonr
from scipy.linalg import orthogonal_procrustes

from sklearn.preprocessing import StandardScaler
from prettytable import PrettyTable

import matplotlib.pyplot as plt
import shap

import os
os.chdir("./src")

from model_utils import models, utils


# generate data assuming an underlying latent space of dimension k
n_train = 50
n_test = 100
n_val = 100
n = n_train + n_val + n_test

x1 = np.random.normal(0., 0.5, size=n)
x2 = np.random.normal(0., 0.5, size=n)
X = np.concatenate([x1[..., None], x2[..., None]], axis=1)
X.shape

x3 = x1**2 + x2**2 + x1*x2
np.median(x3)

# add binary outcome
y = np.zeros(n)
y[x3 > np.median(x3)] = 1.

color_vec = ["red" if ii == 1 else "blue" for ii in y]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(x1, x2, x3, c=color_vec)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()

fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(x1, x2, c=color_vec)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
plt.show()

y = x3 * 1. + np.random.randn(n) * 0.5


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(x1, x2, x3, c=y)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()

fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(x1, x2, c=y)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
plt.show()


# split data
data_split = utils.DataSplit(
    X.shape[0], test_size=n_test, val_size=n_val, scale_features=True, scalers=(StandardScaler(), StandardScaler())
)
print("train: ", len(data_split.train_index), 
    "val: ", len(data_split.val_index),
    "test: ", len(data_split.test_index)
)

data_train = data_split.get_train(X, y)
data_test = data_split.get_test(X, y)
data_val = data_split.get_val(X, y)

# get tensors
data_split = utils.DataSplit(
    X.shape[0], test_size=n_test, val_size=n_val,
    scale_features=True, scalers=(StandardScaler(), StandardScaler()),
    return_tensor=True
)
tensor_data_train = data_split.get_train(X, y[..., None])
tensor_data_test = data_split.get_test(X, y[..., None])
tensor_data_val = data_split.get_val(X, y[..., None])



# Linear model
from sklearn.linear_model import LogisticRegression, LinearRegression

lasso = LinearRegression().fit(data_train[0][:, 0:2], data_train[2])
lasso.coef_

lasso_pred = lasso.predict(data_test[0][:, 0:2])
np.corrcoef(data_test[2], lasso_pred, rowvar=False)**2
np.sqrt(np.mean((lasso_pred - data_test[2])**2))

plt.scatter(data_test[2], lasso_pred)
plt.show()

# with X3
Xnew = np.concat([data_train[0][:, 0:2], data_train[1][..., None]], axis=1)

lasso = LinearRegression().fit(Xnew, y.ravel())
lasso.score(Xnew, y.ravel())

lasso.coef_

lasso_pred = lasso.predict(Xnew)
np.corrcoef(y.ravel(), lasso_pred, rowvar=False)**2
np.sqrt(np.mean((lasso_pred - y.ravel())**2))


# Saturated model - random covariates
lasso = LinearRegression().fit(data_train[0], data_train[2])
lasso.coef_

# on test data
lasso_pred = lasso.predict(data_test[0])
np.corrcoef(data_test[2], lasso_pred, rowvar=False)**2
np.sqrt(np.mean((lasso_pred - data_test[2])**2))

plt.scatter(data_test[2], lasso_pred)
plt.show()

# on train data
lasso_pred = lasso.predict(data_train[0])
np.corrcoef(data_train[2], lasso_pred, rowvar=False)**2
np.sqrt(np.mean((lasso_pred - data_train[2])**2))

plt.scatter(data_train[2], lasso_pred)
plt.show()


# ----------------- with a NN -----------------

# make tensor data loaders
train_dataloader = utils.make_data_loader(*tensor_data_train, batch_size=32)
test_dataloader = utils.make_data_loader(*tensor_data_test, batch_size=32)
val_dataloader = utils.make_data_loader(*tensor_data_val, batch_size=32)


class modelNN(nn.Module):
    def __init__(self, input_dim, latent_dim, dropout_sigma=0.1):
        super(modelNN, self).__init__()
        
        self.dropout = models.GaussianDropout(dropout_sigma)
        self.relu = nn.SiLU()
        
        # Linear outcome prediction
        self.linear = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            self.dropout
        )
        self.hidden = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            self.dropout
        )
        self.out = nn.Linear(latent_dim, 1)

    def forward(self, data):
        x = self.relu(self.linear(data[0]))
        x = self.relu(self.hidden(x))
        # x = self.relu(self.hidden(x))

        y_pred = self.out(x)

        return y_pred

    def predict(self, x):
        x = self.relu(self.linear(x))
        x = self.relu(self.hidden(x))
        # x = self.relu(self.hidden(x))

        y_pred = self.out(x)

        return y_pred

    def loss(self, model_output, data, beta=1.0):
        # label prediction loss
        PredMSE = nn.functional.mse_loss(model_output, data[1], reduction='mean')

        return PredMSE


# 4. Training Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = modelNN(input_dim=X.shape[1], latent_dim=10, dropout_sigma=0.1).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
optimizer = optim.RMSprop(model.parameters())

# 5. Training Loop
num_epochs = 1000
c_annealer = utils.CyclicAnnealer(cycle_length=num_epochs / 2, min_beta=0.0, max_beta=1.0, mode='cosine')

trainer = utils.Training(train_dataloader, val_dataloader)
trainer.losses.keys()

trainer.training_loop(model, optimizer, num_epochs)

plt.plot(trainer.losses["train"], label="train")
plt.plot(trainer.losses["val"], label="val")
plt.legend()
plt.show()

model.eval()
with torch.no_grad():
    y_test_hat = model(tensor_data_test).cpu().numpy()

pearsonr(tensor_data_test[1].numpy().ravel(), y_test_hat.ravel())[0]
np.sqrt(np.mean((y_test_hat.ravel() - tensor_data_test[1].numpy().ravel())**2))

plt.scatter(tensor_data_test[1].numpy().ravel(), y_test_hat.ravel())
plt.show()

# on train
model.eval()
with torch.no_grad():
    y_test_hat = model(tensor_data_train).cpu().numpy()

pearsonr(tensor_data_train[1].numpy().ravel(), y_test_hat.ravel())[0]
np.sqrt(np.mean((y_test_hat.ravel() - tensor_data_train[1].numpy().ravel())**2))

plt.scatter(tensor_data_test[1].numpy().ravel(), y_test_hat.ravel())
plt.show()

# model weights
for name, param in model.named_parameters():
    print(name)
    print(param)


# ---------- DICE --------------
import dice_ml
from dice_ml.utils import helpers
import pandas as pd


class DiceModel(nn.Module):
    def __init__(self, torch_model):
        super(DiceModel, self).__init__()        
        self.torch_model = torch_model

    def forward(self, x):
        return self.torch_model.predict(x)


dice_model = DiceModel(model)

# Initialize DiCE
data = pd.DataFrame(tensor_data_train[0].numpy())
data["y"] = tensor_data_train[1].numpy()
data.columns = ["x0", "x1", "y"]

d = dice_ml.Data(dataframe=data, continuous_features=['x0', 'x1'], outcome_name="y")
m = dice_ml.Model(model=dice_model, backend='PYT', model_type="regressor", func=None)
exp = dice_ml.Dice(d, m)


dice_model(tensor_data_test[0])

# Query instance: We want to change y=1 to y=2
query_instance = data.iloc[0:1].drop(columns="y")

# Generate counterfactuals
cf = exp.generate_counterfactuals(
    query_instance,
    total_CFs=3,  # Number of counterfactuals
    desired_range=[0., 1]  # Target y=2 (exact range)
)

cf.visualize_as_dataframe()


# --------- SHAP ---------
model.eval()
# backgrund samples
background = tensor_data_train[0].numpy()
samples_to_explain = tensor_data_test[0].numpy()

# Define a prediction function
def predict(x):
    x = torch.FloatTensor(x)
    with torch.no_grad():
        preds = model.predict(x)
    return preds.numpy().flatten()

model.predict(torch.FloatTensor(samples_to_explain))
# Create KernelExplainer
explainer = shap.KernelExplainer(predict, background)

samples_to_explain = tensor_data_test[0].numpy()
shap_values = explainer.shap_values(samples_to_explain)
shap_values.shape

# Plot feature importance
feature_names = [f'Feature {i}' for i in range(2)]

shap.summary_plot(shap_values, samples_to_explain, show = True)

shap.summary_plot(shap_values, samples_to_explain, plot_type="violin", show = True)

sample_ind = 1
shap.force_plot(explainer.expected_value, shap_values[sample_ind, :], samples_to_explain[sample_ind, :], 
    feature_names=[f'Feature {i}' for i in range(2)], matplotlib=True
)

# For multiple predictions
shap.force_plot(explainer.expected_value, shap_values, samples_to_explain,
    feature_names=[f'Feature {i}' for i in range(p)], matplotlib=True
)

# dependence plot
shap.dependence_plot(0, shap_values, samples_to_explain, interaction_index=None)

shap.dependence_plot(1, shap_values, samples_to_explain, interaction_index='auto')

shap.dependence_plot(0, shap_values, samples_to_explain, interaction_index=1, cmap='coolwarm')

plt.scatter(shap_values[:, 0], shap_values[:, 1])
plt.show()


sample_ind = 1
shap.plots.waterfall(shap.Explanation(
    values=shap_values[sample_ind], 
    base_values=explainer.expected_value, 
    data=samples_to_explain[sample_ind], 
    feature_names=feature_names
    )
)

shap.plots.heatmap(shap.Explanation(
    values=shap_values, 
    base_values=explainer.expected_value, 
    data=samples_to_explain, 
    feature_names=feature_names
    )
)

shap_interaction = explainer.shap_interaction_values(X)
