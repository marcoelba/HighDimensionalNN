# Example linear regression
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
import matplotlib.pyplot as plt
import shap


# create custom data loader for the training loop
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Define class with simple NN
class NNModel(nn.Module):
    def __init__(self, input_dim: int, ldim: int, output_dim: int = 1):
        super(NNModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, ldim)
        self.fc2 = nn.Linear(ldim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class DeepNNModel(nn.Module):
    def __init__(self, input_dim: int, l1dim: int, l2dim: int, l3dim: int):
        super(DeepNNModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, l1dim)
        self.fc2 = nn.Linear(l1dim, l2dim)
        self.fc3 = nn.Linear(l2dim, l3dim)
        self.fc4 = nn.Linear(l3dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)

        return x


class MixModel(nn.Module):
    def __init__(self, input_dim: int, l_space_dim: int, ldim: int):
        super(MixModel, self).__init__()
        # encoder layers
        self.enc1 = nn.Linear(input_dim, l_space_dim * 2)
        self.enc2 = nn.Linear(l_space_dim * 2, l_space_dim)
        # decoder layers
        self.dec1 = nn.Linear(l_space_dim, l_space_dim * 2)
        self.dec2 = nn.Linear(l_space_dim * 2, input_dim)

        # Fully connected layers
        self.fc1 = nn.Linear(l_space_dim, 1)

    def forward(self, x):
        # encoder-decoder
        x = torch.relu(self.enc1(x))
        x_enc = torch.relu(self.enc2(x))
        x_dec = torch.relu(self.dec1(x_enc))
        x_dec = torch.relu(self.dec2(x_dec))

        # FC
        x = self.fc1(x_enc)

        return x, x_dec


class LinearModel(nn.Module):
    def __init__(self, input_dim: int):
        super(LinearModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = self.fc1(x)

        return x


# reconstruction loss
def decoder_loss(x_true, x_pred):
    return torch.mean((x_true - x_pred)**2)


# R2 score
def r2_score(y_true, y_pred):
    ss_total = torch.sum((y_true - torch.mean(y_true))**2)
    ss_residual = torch.sum((y_true - y_pred)**2)
    return 1 - (ss_residual / ss_total)


class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)


# Generate some data
np.random.seed(134)
n_train = 1000
n_val = 40
n_test = 100
n = n_train + n_val + n_test
p = 100
p1 = 20
p0 = p - p1

X = np.random.normal(size=[n, p]) + np.random.choice([0, 1, 2], size=p)
# beta_true = np.random.choice([-2, -1, 1, 2], size=p)
beta_true = np.concat([np.random.choice([-2, -1, 1, 2], size=p1), np.zeros(p0)])

y = np.dot(X, beta_true) + np.random.normal(size=n) * 0.5
y.shape
y = y[..., np.newaxis]
y.shape


# split train test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=n_test)
X_train.shape

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=n_val)
X_train.shape
X_val.shape

# Convert numpy arrays to PyTorch tensors
X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).float()  # or .long() for class labels
X_test_tensor = torch.from_numpy(X_test).float()
y_test_tensor = torch.from_numpy(y_test).float()
X_val_tensor = torch.from_numpy(X_val).float()
y_val_tensor = torch.from_numpy(y_val).float()  # or .long() for class labels


# Initialize datasets
train_dataset = CustomDataset(X_train_tensor, y_train_tensor)
val_dataset = CustomDataset(X_val_tensor, y_val_tensor)
test_dataset = CustomDataset(X_test_tensor, y_test_tensor)

# crete data loaders
batch_size = 32

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)
val_loader = DataLoader(
    val_dataset, batch_size=round(n_val / 2), shuffle=False
)
test_loader = DataLoader(
    test_dataset, batch_size=round(n_test / 2), shuffle=False  # No shuffle for testing!
)

# Inspect a batch
for batch_X, batch_y in train_loader:
    print(f"Batch X shape: {batch_X.shape}")  # (batch_size, num_features)
    print(f"Batch y shape: {batch_y.shape}")  # (batch_size,) or (batch_size, num_targets)
    break

# Define loss and optimiser and model
model = LinearModel(p)

model = NNModel(p, round(p * np.log(p)))

model = DeepNNModel(p, 1000, 1000, p)

model = MixModel(p, 5, 1)

# test
model(X_test_tensor[1:5, :])[0]
model(X_test_tensor[1:5, :])[1].shape

temp_x, temp_y = next(iter(train_loader))
output, dec = model(temp_x)

criterion(output, temp_y)
decoder_loss(temp_x, dec)


# train loop
# Early stopping
early_stopping = EarlyStopping(patience=20, delta=0.01)

alpha = 1.0

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

n_epochs = 300

train_losses = []
train_decoder_loss = []
train_reg_loss = []

val_losses = []
val_decoder_loss = []
val_reg_loss = []

for epoch in range(n_epochs):
    epoch_train_loss = 0.0
    epoch_reg_loss = 0.0
    epoch_dec_loss = 0.0

    model.train(True)
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output, dec = model(data)
        
        reg_loss = criterion(output, target)
        dec_loss = alpha * decoder_loss(data, dec)
        loss = reg_loss + dec_loss

        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}')
        
        epoch_train_loss += loss.item()
        epoch_reg_loss += reg_loss.item()
        epoch_dec_loss += dec_loss.item()

    # Average epoch loss and store it
    train_losses.append(epoch_train_loss / len(train_loader))
    train_reg_loss.append(epoch_reg_loss / len(train_loader))
    train_decoder_loss.append(epoch_dec_loss / len(train_loader))

    # Validation phase
    model.eval()
    epoch_val_loss = 0.0
    epoch_reg_loss = 0.0
    epoch_dec_loss = 0.0

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            output, dec = model(batch_X)

            reg_loss = criterion(output, batch_y)
            dec_loss = alpha * decoder_loss(batch_X, dec)
            loss = reg_loss + dec_loss

            epoch_val_loss += loss.item()
            epoch_reg_loss += reg_loss.item()
            epoch_dec_loss += dec_loss.item()

    # Average epoch loss and store it
    val_losses.append(epoch_val_loss / len(val_loader))
    val_reg_loss.append(epoch_reg_loss / len(train_loader))
    val_decoder_loss.append(epoch_dec_loss / len(train_loader))

    # early stopping
    # early_stopping(epoch_val_loss / len(val_loader), model)
    # if early_stopping.early_stop:
    #     print("Early stopping")
    #     break

# load best model
early_stopping.load_best_model(model)


# check the loss
plt.plot(train_losses, label='Train Loss', ls="-")
plt.plot(train_reg_loss, label='Train Reg Loss', ls="--")
plt.plot(train_decoder_loss, label='Train Dec Loss', ls="-.")

plt.plot(val_losses, label='Val Loss', ls="-")
plt.plot(val_reg_loss, label='Val Reg Loss', ls="--")
plt.plot(val_decoder_loss, label='Val Dec Loss', ls="-.")

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curves')
plt.legend()
plt.show()

# params
for name, param in model.named_parameters():
    print(name, param)

coeffs = model.fc1.weight.detach().numpy()[0]
coeffs = model.fc2.weight.detach().numpy()[0]

plt.plot(coeffs, color='green', marker='o', linestyle='')
plt.hlines(0, 0, p)
plt.vlines(p1, -1, 1)
plt.show()


# -------------- Compare with LASSO ------------------
lasso = LassoCV(cv=5, random_state=0).fit(X_train, y_train.ravel())
lasso.score(X_train, y_train.ravel())
lasso.score(X_test, y_test.ravel())

plt.plot(lasso.coef_, color='green', marker='o', linestyle='')
plt.hlines(0, 0, p)
plt.vlines(p1, -1, 1)
plt.show()

lasso_pred = lasso.predict(X_test)
np.corrcoef(y_test.ravel(), lasso_pred, rowvar=False)**2

np.sqrt(np.mean((lasso_pred - y_test.ravel())**2))

# model predictions
# 1. Set model to evaluation mode
model.eval()

# 2. Generate predictions (no gradients)
with torch.no_grad():
    test_preds = model(X_test_tensor)[0]

# 3. Calculate RMSE
mse = nn.MSELoss()
rmse = torch.sqrt(mse(test_preds, y_test_tensor)).item()
print(f"RMSE: {rmse:.4f}")

r2 = r2_score(y_test_tensor, test_preds).item()
print(f"R²: {r2:.4f}")

np.corrcoef(y_test_tensor.numpy(), test_preds.numpy(), rowvar=False)**2

# SHAP interpretation
# Wrap the model for SHAP
def model_predict(x):
    # Convert numpy array to torch tensor
    x = torch.from_numpy(x).float()
    # Make prediction
    with torch.no_grad():
        out = model(x)
    # Convert back to numpy array
    return out.numpy()


# Create a SHAP explainer
shap_set = X_train[:50]  # Use a small subset as background
explainer = shap.KernelExplainer(model_predict, shap_set)

# Select instances to explain
test_samples = X_test  # Explain first 5 test samples

# Calculate SHAP values
shap_values = explainer(test_samples)

# Visualize the explanations
fig = shap.force_plot(explainer.expected_value, shap_values[0], test_samples[0], matplotlib=True)

# Summary plot (shows feature importance)
plt.figure()
fig = shap.summary_plot(shap_values, test_samples)
plt.show()

# Dependence plot for a specific feature
shap.dependence_plot(0, shap_values, test_samples)  # First feature
