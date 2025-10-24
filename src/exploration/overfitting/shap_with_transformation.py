# Checking SHAP with transformation
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

import numpy as np
from scipy.stats import pearsonr
from scipy.special import softplus
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import shap

from src.utils import training_wrapper
from src.utils import data_loading_wrappers
from src.utils.model_output_details import count_parameters
from src.utils import plots


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
        # self.dropout = nn.Dropout(p=dropout_prob)
        self.gelu = nn.GELU()

    def forward(self, batch):
        x = self.gelu(self.fc1(batch[0]))
        pred = self.fc2(x)

        return pred

    def loss(self, m_out, batch):
        # label prediction loss
        return nn.functional.mse_loss(m_out, batch[1], reduction='sum')


def softplus_inverse(y):
    return np.log(np.exp(y) - 1)

# generate data
n_train = 100
n_test = 200
n_val = 200
n = n_train + n_val + n_test
p = 10
batch_size = 50

# custom W
np.random.seed(323425)
W = np.random.choice(
    [-1, -0.8, -0.5, 1, 0.8, 0.5],
    size=(p)
)
mean_covariates = np.random.choice([-1, -0.5, 0, 0.5, 1.], size=p)
var_covariates = np.random.choice([0.1, 1., 2.], size=p)

X = np.random.randn(n, p)
np.round(np.cov(X, rowvar=False), 2)
cov_matrix = np.ones([p, p]) * 0.
np.fill_diagonal(cov_matrix, var_covariates)
X = X.dot(np.linalg.cholesky(cov_matrix).transpose())
np.round(np.cov(X, rowvar=False), 2)
y = np.dot(X, W) + np.random.randn(n) * 0.6
y = softplus(y)
y = y[..., None]

plt.hist(y)
plt.show()

plt.hist(softplus_inverse(y))
plt.show()


# ------------------ OLS ---------------------
W
np.linalg.inv(np.transpose(X).dot(X)).dot(np.transpose(X).dot(y)).squeeze()
# with standardised features
X_scales = X.std(axis=0)
X_std = X / X_scales
X_std.std(axis=0)

beta_std = np.linalg.inv(np.transpose(X_std).dot(X_std)).dot(np.transpose(X_std).dot(y)).squeeze()
beta_std / X_scales
# -------------------------------------------------


# we need y on the real axis
y = softplus_inverse(y)

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

# features and outcome preprocessing
scaler_cov = StandardScaler()
scaler_cov.fit(X=tensor_data_train[0])
scaler_cov.mean_
scaler_cov.scale_

scaler_outcome = StandardScaler()
scaler_outcome.fit(X=tensor_data_train[1])
scaler_outcome.mean_
scaler_outcome.scale_

class TorchScaler:
    def __init__(self, scaler, dtype=torch.float32):
        self.mean = torch.tensor(scaler.mean_, dtype=dtype)
        self.scale = torch.tensor(scaler.scale_, dtype=dtype)

    def transform(self, x):
        """
            x: torch.Tensor
        """
        return (x - self.mean) / self.scale

    def inverse_transform(self, x):
        """
            x: torch.Tensor
        """
        return x * self.scale + self.mean


torch_scaler_outcome = TorchScaler(scaler_outcome)
torch_scaler_outcome.mean
torch_scaler_outcome.scale
torch_scaler_outcome.transform(tensor_data_train[1])
torch_scaler_outcome.inverse_transform(tensor_data_train[1])

torch_scaler_cov = TorchScaler(scaler_cov)
torch_scaler_cov.mean
torch_scaler_cov.scale
torch_scaler_cov.transform(tensor_data_train[0])
tensor_data_train[0][0:3] * torch_scaler_cov.scale + torch_scaler_cov.mean
scaler_cov.inverse_transform(tensor_data_train[0][0:3])

# apply scaling
preproc_tensor_data_train = [
    torch.tensor(scaler_cov.transform(tensor_data_train[0]), dtype=torch.float32),
    torch.tensor(scaler_outcome.transform(tensor_data_train[1]), dtype=torch.float32)
]
preproc_tensor_data_val = [
    torch.tensor(scaler_cov.transform(tensor_data_val[0]), dtype=torch.float32),
    torch.tensor(scaler_outcome.transform(tensor_data_val[1]), dtype=torch.float32)
]
preproc_tensor_data_test = [
    torch.tensor(scaler_cov.transform(tensor_data_test[0]), dtype=torch.float32),
    torch.tensor(scaler_outcome.transform(tensor_data_test[1]), dtype=torch.float32)
]

# make tensor data loaders
preproc_train_dataloader = data_loading_wrappers.make_data_loader(*preproc_tensor_data_train, batch_size=batch_size)
preproc_test_dataloader = data_loading_wrappers.make_data_loader(*preproc_tensor_data_test, batch_size=batch_size)
preproc_val_dataloader = data_loading_wrappers.make_data_loader(*preproc_tensor_data_val, batch_size=batch_size)


# 4. Training Setup
device = torch.device("cpu")
model_nn = NNModel(p, ldim=10, dropout_prob=0.2).to(device)
optimizer = optim.Adam(model_nn.parameters(), lr=1e-3)

# Coefficients
count_parameters(model_nn)

# Training Loop
num_epochs = 200

trainer = training_wrapper.Training(preproc_train_dataloader, preproc_val_dataloader)
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

# SHAP explanations
class TensorModel(torch.nn.Module):
    def __init__(self, model):
        super(TensorModel, self).__init__()
        self.model = model
    
    def forward(self, x):
        """
        Args:
            x: torch tensor array with ALL features concatenated
        """
        x_list = [x]
        output = self.model(x_list)
        return output


model_shap = TensorModel(model_nn)
background_data = preproc_tensor_data_train[0]
model_shap(background_data)
explainer = shap.GradientExplainer(model_shap, background_data)
explain_data = preproc_tensor_data_test[0]

shap_values = explainer.shap_values(explain_data, return_variances=False)
shap_values.shape
shap_values = shap_values[..., -1]

W
shap.summary_plot(shap_values, explain_data, show = True)

# Get base values for the explainer
model_shap.eval()
with torch.no_grad():
    predictions_background = model_shap(background_data)
    base_value = predictions_background.numpy().mean()

feat_names = [f"f_{ii}" for ii in range(p)]

# model predictions on test data
model_shap.eval()
with torch.no_grad():
    predictions = model_shap(explain_data).numpy()

# Create Explanation object manually
ind = 1
explanation = shap.Explanation(
    values=shap_values[ind],  # For single output
    base_values=base_value,
    data=explain_data[ind].numpy(),  # Flatten if needed
    feature_names=feat_names
)

base_value + np.sum(shap_values[ind])

W
base_value
predictions[ind]
explain_data[ind]
shap.plots.waterfall(explanation)

sum(shap_values[ind] / scaler_cov.scale_)

# -----------------------------------------------------------
# Explanations on original outcome
def torch_softplus(y):
    m = nn.Softplus()
    return m(y)

torch_scaler_cov = TorchScaler(scaler_cov)
torch_scaler_outcome = TorchScaler(scaler_outcome)

class TensorModelRescaled(torch.nn.Module):
    def __init__(self, model, torch_scaler_outcome=None, torch_scaler_cov=None):
        super(TensorModelRescaled, self).__init__()
        self.model = model
        self.torch_scaler_outcome = torch_scaler_outcome
        self.torch_scaler_cov = torch_scaler_cov

    def forward(self, x):
        """
        Args:
            x: torch tensor with features in the ORIGINAL scale
        """
        if self.torch_scaler_cov is not None:
            x = self.torch_scaler_cov.transform(x)

        output = self.model([x])
        
        # rescale outcome using sklearn object
        if self.torch_scaler_outcome is not None:
            output = self.torch_scaler_outcome.inverse_transform(output)
        # inverse-log transformation
        output = torch_softplus(output)
        return output

model_shap_rescaled = TensorModelRescaled(
    model_nn,
    torch_scaler_outcome=torch_scaler_outcome,
    torch_scaler_cov=torch_scaler_cov
)
background_data = tensor_data_train[0]
model_shap_rescaled(background_data)

explainer_rescaled = shap.GradientExplainer(model_shap_rescaled, background_data)
explain_data = tensor_data_test[0]

shap_values_rescaled = explainer_rescaled.shap_values(explain_data, return_variances=False)
shap_values_rescaled.shape
shap_values_rescaled = shap_values_rescaled[..., -1]

W
shap.summary_plot(shap_values_rescaled, explain_data, show = True)

# get base value
model_shap_rescaled.eval()
with torch.no_grad():
    base_value_rescaled = model_shap_rescaled(background_data).numpy().mean()

# get predictions
model_shap_rescaled.eval()
with torch.no_grad():
    predictions_rescaled = model_shap_rescaled(explain_data).numpy()

ind = 0
explanation_rescaled = shap.Explanation(
    values=shap_values_rescaled[ind],  # For single output
    base_values=base_value_rescaled,
    data=explain_data[ind].numpy(),  # Flatten if needed
    feature_names=feat_names
)

base_value_rescaled + sum(shap_values_rescaled[ind])

W
base_value_rescaled
predictions_rescaled[ind]
background_data[ind]
shap.plots.waterfall(explanation_rescaled)


# NO Rescaling for features but post rescaling
model_shap_rescaled = TensorModelRescaled(
    model_nn,
    torch_scaler_outcome=torch_scaler_outcome,
    torch_scaler_cov=None
)
background_data = preproc_tensor_data_train[0]
model_shap_rescaled(background_data)

explainer_rescaled = shap.GradientExplainer(model_shap_rescaled, background_data)
explain_data = preproc_tensor_data_test[0]

shap_values_rescaled = explainer_rescaled.shap_values(explain_data, return_variances=False)
shap_values_rescaled.shape
shap_values_rescaled = shap_values_rescaled[..., -1]

W
shap.summary_plot(shap_values_rescaled, explain_data, show = True)
# rescale using features scales
shap_values_rescaled = shap_values_rescaled / scaler_cov.scale_

# get base value
model_shap_rescaled.eval()
with torch.no_grad():
    base_value_rescaled = model_shap_rescaled(background_data).numpy().mean()

# get predictions
model_shap_rescaled.eval()
with torch.no_grad():
    predictions_rescaled = model_shap_rescaled(explain_data).numpy()

ind = 0
explanation_rescaled = shap.Explanation(
    values=shap_values_rescaled[ind],  # For single output
    base_values=base_value_rescaled,
    data=tensor_data_test[0][ind].numpy(),  # Flatten if needed
    feature_names=feat_names
)

base_value_rescaled + sum(shap_values_rescaled[ind])

W
base_value_rescaled
predictions_rescaled[ind]
background_data[ind]
shap.plots.waterfall(explanation_rescaled)



class GradientExplainer:
    def __init__(self, model, background_data):
        """
        model: PyTorch model which gets list of tensors as input
        background_data: torch.Tensor of shape (n_background, n_features)
        """
        self.model = model
        self.background_data = background_data
        self.expected_value = self._compute_expected_value()
    
    def _compute_expected_value(self):
        """Compute average model prediction over background data"""

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(self.background_data)
        return predictions.mean().item()
    
    def shap_values(self, X, n_samples_path=50, n_baselines=10):
        """
        X: list of torch.Tensor of instances to explain
        n_samples_path: number of points to sample along the path
        n_baselines: number of baseline samples to use
        """
        if n_baselines == 0:
            n_baselines = self.background_data[0].shape[0]
        
        n_instances, n_features = X.shape
        shap_values = [torch.zeros_like(X)]
        
        # Sample multiple baselines
        baseline_indices = torch.randint(0, len(self.background_data), (n_baselines,))
        baselines = self.background_data[baseline_indices]
        
        for i in range(n_instances):
            instance_shap = self._explain_instance(X[i], baselines, n_samples_path)
            shap_values[i] = instance_shap
            
        return shap_values.detach().numpy()
    
    def _explain_instance(self, instance, baselines, n_samples):
        """Explain a single instance"""
        n_features = instance.shape[0]
        n_baselines = len(baselines)
        instance_shap = torch.zeros(n_features)
        
        # Sample points along the path for each baseline
        for baseline in baselines:
            # Sample alpha values (path parameter)
            alphas = torch.rand(n_samples)
            
            for alpha in alphas:
                # Create interpolated point
                point = baseline + alpha * (instance - baseline)
                point = point.requires_grad_(True)
                
                # Forward pass
                prediction = self.model(point)
                
                # Backward pass - compute gradients
                prediction.backward()
                
                # Accumulate gradients
                with torch.no_grad():
                    instance_shap += (instance - baseline) * point.grad
                
                # Zero gradients for next iteration
                self.model.zero_grad()
        
        # Average over all baselines and path samples
        instance_shap /= (n_baselines * n_samples)
        return instance_shap



background_data = tensor_data_train[0]
explain_data = tensor_data_test[0]

explainer = GradientExplainer(model_nn, background_data)
explainer.expected_value
shap_values = explainer.shap_values(explain_data, n_samples_path=100, n_baselines=20)



# test # 

class GradientExplainer:
    def __init__(self, model, background_data):
        """
        model: PyTorch model
        background_data: torch.Tensor of shape (n_background, n_features)
        """
        self.model = model
        self.background_data = background_data
        self.expected_value = self._compute_expected_value()
    
    def _compute_expected_value(self):
        """Compute average model prediction over background data"""
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(self.background_data)
        return predictions.mean().item()
    
    def shap_values(self, X, n_samples=50, n_baselines=10):
        """
        X: torch.Tensor of instances to explain (shape: n_instances x n_features)
        n_samples: number of points to sample along the path
        n_baselines: number of baseline samples to use
        """
        if len(X.shape) == 1:
            X = X.unsqueeze(0)
            
        n_instances, n_features = X.shape
        shap_values = torch.zeros_like(X)
        
        # Sample multiple baselines
        baseline_indices = torch.randint(0, len(self.background_data), (n_baselines,))
        baselines = self.background_data[baseline_indices]
        
        for i in range(n_instances):
            instance_shap = self._explain_instance(X[i], baselines, n_samples)
            shap_values[i] = instance_shap
            
        return shap_values.detach().numpy()
    
    def _explain_instance(self, instance, baselines, n_samples):
        """Explain a single instance"""
        n_features = instance.shape[0]
        n_baselines = len(baselines)
        instance_shap = torch.zeros(n_features)
        grad_norms = []
        
        self.model.eval()
        # Sample points along the path for each baseline
        for baseline in baselines:
            # Sample alpha values (path parameter)
            alphas = torch.rand(n_samples)
            
            for alpha in alphas:
                # Create interpolated point
                point = baseline + alpha * (instance - baseline)
                point = point.requires_grad_(True)
                
                # Forward pass
                prediction = self.model(point)
                
                # Backward pass - compute gradients
                prediction.backward()
                
                # Debug: Check gradient magnitude
                grad_norm = point.grad.norm().item()
                grad_norms.append(grad_norm)
                
                if point.grad is None:
                    print("WARNING: No gradients computed!")
                    continue

                # Accumulate gradients
                with torch.no_grad():
                    instance_shap += (instance - baseline) * point.grad
                
                # Zero gradients for next iteration
                self.model.zero_grad()
        
        # Average over all baselines and path samples
        instance_shap /= (n_baselines * n_samples)
        return instance_shap


background_data = tensor_data_train[0]
explain_data = tensor_data_test[0][0:5]

class TensorModel(torch.nn.Module):
    def __init__(self, model):
        super(TensorModel, self).__init__()
        self.model = model
    
    def forward(self, x):
        """
        Args:
            x: torch tensor array with ALL features concatenated
        """
        output = self.model([x])
        return output

model = TensorModel(model_nn)
model_nn.eval()
model_nn([background_data]).mean()

explainer = GradientExplainer(model, background_data)
explainer.expected_value

shap_values = explainer.shap_values(explain_data, n_samples=100, n_baselines=100)
shap_values.shape

model_nn.eval()
model_nn([explain_data])
model(explain_data)
np.round(explainer.expected_value + shap_values.sum(axis=1), 2)


def model_predct_shap(x):
    model_nn.eval()
    with torch.no_grad():
        output = model_nn([torch.tensor(x)]).numpy()
    return output

model_predct_shap(background_data)
explainer_kernel = shap.KernelExplainer(model_predct_shap, background_data.numpy())
explainer_kernel.expected_value

shap_values_kernel = explainer_kernel.shap_values(explain_data.numpy())
shap_values_kernel = shap_values_kernel[..., -1]

explainer_kernel.expected_value + shap_values_kernel.sum(axis=1)



def check_gradient_behavior(model, background_data, test_instances):
    """Check if gradients are well-behaved"""
    model.eval()
    
    # Sample some points along the interpolation path
    baseline = background_data[0:1]
    instance = test_instances[0:1]
    
    alphas = torch.linspace(0, 1, 10)
    gradients = []
    predictions = []
    
    for alpha in alphas:
        point = baseline + alpha * (instance - baseline)
        point = point.requires_grad_(True)
        
        pred = model(point)
        pred.backward()
        
        gradients.append(point.grad.norm().item())
        predictions.append(pred.item())
        
        model.zero_grad()
    
    print("Gradient norms along path:", [f"{g:.6f}" for g in gradients])
    print("Predictions along path:", [f"{p:.6f}" for p in predictions])
    
    # Check for problematic patterns
    if max(gradients) < 1e-6:
        print("🚨 WARNING: Gradients are extremely small!")
    if any(np.isnan(g) for g in gradients):
        print("🚨 WARNING: NaN gradients detected!")

check_gradient_behavior(model, background_data, explain_data)


def comprehensive_gradient_check(model, instance, background_data, n_checks=5):
    """Thorough gradient analysis"""
    model.eval()
    # model.dropout.eval()  # Explicitly disable dropout
    
    print("=== COMPREHENSIVE GRADIENT CHECK ===")
    
    for i in range(min(n_checks, len(background_data))):
        baseline = background_data[i:i+1]
        
        # Test multiple points along the path
        for alpha in [0.1, 0.5, 0.9]:
            point = baseline + alpha * (instance - baseline)
            point = point.requires_grad_(True)
            
            # Forward pass
            pred = model([point])
            
            # Backward pass
            model.zero_grad()
            pred.backward()
            
            grad = point.grad
            print(f"Baseline {i}, alpha {alpha}: "
                  f"pred={pred.item():.4f}, "
                  f"grad_norm={grad.norm().item():.6f}, "
                  f"grad_mean_abs={grad.abs().mean().item():.6f}")
            
            # Check for NaN/inf
            if torch.isnan(grad).any() or torch.isinf(grad).any():
                print("🚨 NaN or Inf in gradients!")

# Run the check
comprehensive_gradient_check(model_nn, explain_data[0], background_data)
