# Model with missing key predictors
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

import numpy as np
from scipy.stats import pearsonr

import matplotlib.pyplot as plt
import shap

from src.utils import training_wrapper
from src.utils import data_loading_wrappers
from src.utils.model_output_details import count_parameters
from src.utils import plots


torch.get_num_threads()
torch.set_num_threads(6)


class LinearModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 1):
        super(LinearModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)

    def forward(self, batch):
        return self.fc1(batch[0])

    def loss(self, m_out, batch):
        # label prediction loss
        return nn.functional.mse_loss(m_out, batch[1], reduction='sum')


class NNModel(nn.Module):
    def __init__(self, input_dim: int, ldim: int, dropout_prob, output_dim: int = 1):
        super(NNModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, ldim)
        self.fc2 = nn.Linear(ldim, output_dim)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, batch):
        x = torch.relu(self.fc1(batch[0]))
        x = self.dropout(x)
        pred = self.fc2(x)

        return pred

    def loss(self, m_out, batch):
        # label prediction loss
        return nn.functional.mse_loss(m_out, batch[1], reduction='sum')


# generate simple linear data
n_train = 100
n_test = 200
n_val = 200
n = n_train + n_val + n_test
p = 10
p1 = 8
p0 = 2
batch_size = 50

# custom W
np.random.seed(323425)
W = np.random.choice(
    [-1, -0.8, 1, 0.8],
    size=(p)
)

X = np.random.randn(n, p1)
np.round(np.cov(X, rowvar=False), 2)
cov_matrix = np.ones([p, p]) * 0.6
np.fill_diagonal(cov_matrix, 1.)
X = X.dot(np.linalg.cholesky(cov_matrix).transpose())
np.round(np.cov(X, rowvar=False), 2)
y = np.dot(X[:, 0:p1], W) + np.dot(X[:, p1:p], beta) + np.random.randn(n) * 0.5
y = y[..., None]

# get tensors
X_tensor = torch.FloatTensor(X).to(torch.device("cpu"))
y_tensor = torch.FloatTensor(y).to(torch.device("cpu"))

# split data
data_split = data_loading_wrappers.DataSplit(X.shape[0], test_size=n_test, val_size=n_val)
print("train: ", len(data_split.train_index),
    "val: ", len(data_split.val_index),
    "test: ", len(data_split.test_index)
)

tensor_data_train = data_split.get_train(X_tensor, y_tensor)
tensor_data_test = data_split.get_test(X_tensor, y_tensor)
tensor_data_val = data_split.get_val(X_tensor, y_tensor)

# make tensor data loaders
train_dataloader = data_loading_wrappers.make_data_loader(*tensor_data_train, batch_size=batch_size)
test_dataloader = data_loading_wrappers.make_data_loader(*tensor_data_test, batch_size=batch_size)
val_dataloader = data_loading_wrappers.make_data_loader(*tensor_data_val, batch_size=batch_size)

next(iter(train_dataloader))[0].shape  # X
next(iter(train_dataloader))[1].shape  # y


# 4. Training Setup
device = torch.device("cpu")

model_nn = NNModel(p, ldim=50, dropout_prob=0.2).to(device)
optimizer = optim.Adam(model_nn.parameters(), lr=1e-3)

# Coefficients
count_parameters(model_nn)

# Training Loop
num_epochs = 500

trainer = training_wrapper.Training(train_dataloader, val_dataloader)
trainer.training_loop(model_nn, optimizer, num_epochs)

plt.plot(trainer.losses["train"], label="train")
plt.plot(trainer.losses["val"], label="val")
plt.vlines(np.argmin(trainer.losses["val"]), 0, max(trainer.losses["val"]), color="red")
plt.vlines(np.argmin(trainer.losses["train"]), 0, max(trainer.losses["train"]), color="blue")
plt.hlines(np.min(trainer.losses["val"]), 0, len(trainer.losses["val"]), color="red", linestyles="--")
plt.hlines(np.min(trainer.losses["train"]), 0, len(trainer.losses["val"]), color="blue", linestyles="--")
plt.legend()
plt.show()
# -----------------------
# NO OVERFITTING

model_linear = LinearModel(p).to(device)
optimizer = optim.RMSprop(model_linear.parameters(), lr=1e-2)
# Training Loop
num_epochs = 300

trainer = training_wrapper.Training(train_dataloader, val_dataloader)
trainer.training_loop(model_linear, optimizer, num_epochs)

plt.plot(trainer.losses["train"], label="train")
plt.plot(trainer.losses["val"], label="val")
plt.vlines(np.argmin(trainer.losses["val"]), 0, max(trainer.losses["val"]), color="red")
plt.vlines(np.argmin(trainer.losses["train"]), 0, max(trainer.losses["train"]), color="blue")
plt.hlines(np.min(trainer.losses["val"]), 0, len(trainer.losses["val"]), color="red", linestyles="--")
plt.hlines(np.min(trainer.losses["train"]), 0, len(trainer.losses["val"]), color="blue", linestyles="--")
plt.legend()
plt.show()

model_linear.fc1.weight

# Compare the predictive performance on test set
model_nn.eval()
model_linear.eval()
with torch.no_grad():
    pred_nn = model_nn(tensor_data_test)
    pred_lin = model_linear(tensor_data_test)

nn.functional.mse_loss(pred_nn, tensor_data_test[1], reduction='mean')
nn.functional.mse_loss(pred_lin, tensor_data_test[1], reduction='mean')
# correlation
pearsonr(pred_nn, tensor_data_test[1])
pearsonr(pred_lin, tensor_data_test[1])
plt.scatter(x=pred_nn, y=tensor_data_test[1])
plt.scatter(x=pred_lin, y=tensor_data_test[1])
plt.show()


# Use SHAP to determine feature importance
class TensorModel(torch.nn.Module):
    def __init__(self, model):
        super(TensorModel, self).__init__()
        self.model = model
        
    def forward(self, *x):
        """
        Args:
            x: torch tensor array with ALL features concatenated
        """
        x_list = list(x)
        output = self.model(x_list)
        return output


model_shap = TensorModel(model_nn)
background_data = tensor_data_train[0:1]
model_shap(*background_data)
explainer = shap.GradientExplainer(model_shap, background_data)

shap_values = explainer.shap_values(background_data)
shap_values.shape
shap_values = shap_values[..., -1]

shap.summary_plot(shap_values, background_data[0], show = True)

shap.summary_plot(shap_values, background_data[0], plot_type="bar")

# Get base values for the explainer
model_shap.eval()
with torch.no_grad():
    predictions = model_shap(*background_data)
    base_value = predictions.numpy().mean()

feat_names = [f"f_{ii}" for ii in range(p)]

# Create Explanation object manually
ind = 0
explanation = shap.Explanation(
    values=shap_values[ind],  # For single output
    base_values=base_value,
    data=background_data[ind][0].numpy(),  # Flatten if needed
    feature_names=feat_names
)

[W, beta]
background_data[ind][0]
shap.plots.waterfall(explanation)

# -------------------------------------------------------
# ----------- Now removing 2 key predictors -------------
# -------------------------------------------------------

X1 = X[:, 0:p1]
X1.shape

X_tensor = torch.FloatTensor(X1).to(torch.device("cpu"))

# split data
data_split = data_loading_wrappers.DataSplit(X1.shape[0], test_size=n_test, val_size=n_val)
print("train: ", len(data_split.train_index), 
    "val: ", len(data_split.val_index),
    "test: ", len(data_split.test_index)
)
tensor_data_train = data_split.get_train(X_tensor, y_tensor)
tensor_data_test = data_split.get_test(X_tensor, y_tensor)
tensor_data_val = data_split.get_val(X_tensor, y_tensor)

# make tensor data loaders
train_dataloader = data_loading_wrappers.make_data_loader(*tensor_data_train, batch_size=batch_size)
test_dataloader = data_loading_wrappers.make_data_loader(*tensor_data_test, batch_size=batch_size)
val_dataloader = data_loading_wrappers.make_data_loader(*tensor_data_val, batch_size=batch_size)


# 4. Training Setup
device = torch.device("cpu")

model1_nn = NNModel(p1, ldim=50, dropout_prob=0.2).to(device)
optimizer = optim.Adam(model1_nn.parameters(), lr=1e-3)
# Coefficients
count_parameters(model1_nn)

# Training Loop
num_epochs = 500

trainer = training_wrapper.Training(train_dataloader, val_dataloader, noisy_gradient=False)
trainer.training_loop(model1_nn, optimizer, num_epochs, gradient_noise_std=0.0)

plt.plot(trainer.losses["train"], label="train")
plt.plot(trainer.losses["val"], label="val")
plt.vlines(np.argmin(trainer.losses["val"]), 0, max(trainer.losses["val"]), color="red")
plt.vlines(np.argmin(trainer.losses["train"]), 0, max(trainer.losses["train"]), color="blue")
plt.hlines(np.min(trainer.losses["val"]), 0, len(trainer.losses["val"]), color="red", linestyles="--")
plt.hlines(np.min(trainer.losses["train"]), 0, len(trainer.losses["val"]), color="blue", linestyles="--")
plt.legend()
plt.show()

# use the model at the best validation iteration
model1_nn.load_state_dict(trainer.best_model.state_dict())


model1_linear = LinearModel(p1).to(device)
optimizer = optim.Adam(model1_linear.parameters(), lr=1e-2)
# Coefficients
count_parameters(model1_linear)

# Training Loop
num_epochs = 300

trainer = training_wrapper.Training(train_dataloader, val_dataloader)
trainer.training_loop(model1_linear, optimizer, num_epochs)

plt.plot(trainer.losses["train"], label="train")
plt.plot(trainer.losses["val"], label="val")
plt.vlines(np.argmin(trainer.losses["val"]), 0, max(trainer.losses["val"]), color="red")
plt.vlines(np.argmin(trainer.losses["train"]), 0, max(trainer.losses["train"]), color="blue")
plt.hlines(np.min(trainer.losses["val"]), 0, len(trainer.losses["val"]), color="red", linestyles="--")
plt.hlines(np.min(trainer.losses["train"]), 0, len(trainer.losses["val"]), color="blue", linestyles="--")
plt.legend()
plt.show()

model1_linear.fc1.weight

# ----------------------------------------------
# Compare the predictive performance on test set
model1_nn.eval()
model1_linear.eval()
with torch.no_grad():
    pred_nn = model1_nn(tensor_data_test)
    pred_lin = model1_linear(tensor_data_test)

nn.functional.mse_loss(pred_nn, tensor_data_test[1], reduction='mean')
nn.functional.mse_loss(pred_lin, tensor_data_test[1], reduction='mean')
# correlation
pearsonr(pred_nn, tensor_data_test[1])
pearsonr(pred_lin, tensor_data_test[1])

# Use SHAP to determine feature importance
class TensorModel(torch.nn.Module):
    def __init__(self, model):
        super(TensorModel, self).__init__()
        self.model = model
        
    def forward(self, x):
        """
        Args:
            x: torch tensor array with ALL features concatenated
        """
        # Transform input back to a list of tensors
        tensors_list = []
        tensors_list.append(x[:, 0:-1])
        tensors_list.append(x[:, -1:])
        output = self.model(tensors_list)
        return output


model_shap = TensorModel(model1_nn)
background_data = torch.concat(tensor_data_train, dim=1)
model_shap(background_data)
print("Shape background_data for SHAP: ", background_data.shape)
explainer = shap.GradientExplainer(model_shap, background_data)

shap_values = explainer.shap_values(background_data)
shap_values.shape
shap_values = shap_values[:, 0:p1, 0]

shap.summary_plot(shap_values, background_data[:, 0:p1], show = True)

shap.plots.heatmap(shap.Explanation(
    values=shap_values, 
    base_values=explainer.expected_value[latent_axis], 
    data=samples_to_explain, 
    feature_names=feature_names
    )
)

#############
n = 100
x = np.random.beta(1, 5, n)
plt.hist(x)
plt.show()

x_std = (x - x.mean()) / x.std()
plt.hist(x_std)
plt.show()



#############
