# test Neural Network
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.model_selection import train_test_split


# Define class with simple NN
class Net(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 5)  # Input layer
        self.fc2 = nn.Linear(5, 5)     # Hidden layer
        self.fc3 = nn.Linear(5, output_dim)      # Output layer (10 classes)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Generate some data
np.random.seed(134)
n = 1000
p = 20

X = np.random.normal(size=[n, p])
beta_true = np.random.choice([-2, -1, 1, 2], size=p)

y = np.dot(X, beta_true) + np.random.normal(size=n) * 0.5
y.shape
y = y[..., np.newaxis]
y.shape


# split train test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Convert numpy arrays to PyTorch tensors
X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).float()  # or .long() for class labels
X_test_tensor = torch.from_numpy(X_test).float()
y_test_tensor = torch.from_numpy(y_test).float()


# create custom data loader for the training loop
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Initialize datasets
train_dataset = CustomDataset(X_train_tensor, y_train_tensor)
test_dataset = CustomDataset(X_test_tensor, y_test_tensor)
#
train_dataset.__getitem__(1)


# crete data loaders
batch_size = 32

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False  # No shuffle for testing!
)

# Inspect a batch
for batch_X, batch_y in train_loader:
    print(f"Batch X shape: {batch_X.shape}")  # (batch_size, num_features)
    print(f"Batch y shape: {batch_y.shape}")  # (batch_size,) or (batch_size, num_targets)
    break

# Define loss and optimiser and model
model = Net(p, 1)
model.parameters()
model(X_train_tensor[0:5, :])

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# train loop
n_epochs = 50
for epoch in range(n_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}')


# model predictions
# 1. Set model to evaluation mode
model.eval()

# 2. Generate predictions (no gradients)
with torch.no_grad():
    test_preds = model(X_test_tensor)

# 3. Calculate RMSE
mse = nn.MSELoss()
rmse = torch.sqrt(mse(test_preds, y_test_tensor)).item()
print(f"RMSE: {rmse:.4f}")

# R2 score
def r2_score(y_true, y_pred):
    ss_total = torch.sum((y_true - torch.mean(y_true))**2)
    ss_residual = torch.sum((y_true - y_pred)**2)
    return 1 - (ss_residual / ss_total)

r2 = r2_score(y_test_tensor, test_preds).item()
print(f"R²: {r2:.4f}")

np.corrcoef(y_test_tensor.numpy(), test_preds.numpy(), rowvar=False)**2
