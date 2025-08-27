import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.nn.utils.parametrizations import orthogonal
import numpy as np
from scipy.linalg import toeplitz
from scipy.stats import multivariate_normal
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from prettytable import PrettyTable


class DataSplit:
    def __init__(self, n, test_size, val_size=None, unique_ids=None, scale_features=False, scalers=None, return_tensor=False):

        train_index, test_index = train_test_split(range(n), test_size=test_size)

        if val_size is not None:
            train_index, val_index = train_test_split(train_index, test_size=val_size)
            self.do_validation = True
        else:
            self.do_validation = False
        
        # if ids array is provided, take the rows indexes from there
        if unique_ids is not None:
            self.train_index = np.where(np.isin(unique_ids, train_index))[0]
            self.test_index = np.where(np.isin(unique_ids, test_index))[0]
            if self.do_validation:
                self.val_index = np.where(np.isin(unique_ids, val_index))[0]
        else:
            self.train_index = train_index
            self.test_index = test_index
            if self.do_validation:
                self.val_index = val_index
        
        # feature scaling
        self.scale_features = scale_features
        self.scalers = scalers
        self.return_tensor = return_tensor

    def get_train(self, *arrays):
        data = []
        for ii, arr in enumerate(arrays):
            dd = arr[self.train_index]

            if self.scale_features:
                self.scalers[ii].fit(dd)
                dd = self.scalers[ii].transform(dd)
            
            if self.return_tensor:
                dd = torch.Tensor(dd)
            
            data.append(dd)
        return data

    def get_test(self, *arrays):
        data = []
        for ii, arr in enumerate(arrays):
            dd = arr[self.test_index]

            if self.scale_features:
                dd = self.scalers[ii].transform(dd)

            if self.return_tensor:
                dd = torch.Tensor(dd)

            data.append(dd)
        return data

    def get_val(self, *arrays):
        data = []
        for ii, arr in enumerate(arrays):
            dd = arr[self.val_index]

            if self.scale_features:
                dd = self.scalers[ii].transform(dd)

            if self.return_tensor:
                dd = torch.Tensor(dd)

            data.append(dd)
        return data


def make_data_loader(*arrays, batch_size = 32, device=torch.device("cpu")):
    # Initialize datasets
    dataset = TensorDataset(*arrays)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    return loader


class LongitudinalDataset(Dataset):
    def __init__(self, X_long, y_long, measurement_ids, time_steps, device=torch.device("cpu")):
        """
        Args:
            X_long: (n_samples, p) - Features in long format (may contain NaNs)
            y_long: (n_samples, T) - Torch tensor, Outcomes in long format (may contain NaNs)
            measurement_ids: (n_samples,) - IDs for measurement types (0 to M-1)
            time_steps: Number of time points (T)
        """
        self.device = device

        self.X = torch.from_numpy(X_long).float()
        self.y = torch.from_numpy(y_long).float()
        self.measurement_ids = torch.from_numpy(measurement_ids).long()
        self.time_steps = time_steps
        
        # Masks for observed data
        self.X_mask = ~torch.isnan(self.X[:, 0])  # (n_samples,)
        self.y_mask = ~torch.isnan(self.y[:, 0])   # (n_samples,)
        
        # Replace NaNs with 0 (will be masked)
        self.X[torch.isnan(self.X)] = 0
        self.y[torch.isnan(self.y)] = 0

        self.X.to(device)
        self.y.to(device)
        self.X_mask.to(device)
        self.y_mask.to(device)
        self.measurement_ids.to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            'X': self.X[idx],
            'y': self.y[idx],
            'measurement_id': self.measurement_ids[idx],
            'X_mask': self.X_mask[idx],  # 1=observed, 0=missing
            'y_mask': self.y_mask[idx]    # 1=observed, 0=missing
        }


def collate_fn(batch):
    # Stack all items in the batch
    X = torch.stack([item['X'] for item in batch])
    y = torch.stack([item['y'] for item in batch])
    measurement_ids = torch.stack([item['measurement_id'] for item in batch])
    X_mask = torch.stack([item['X_mask'] for item in batch])
    y_mask = torch.stack([item['y_mask'] for item in batch])
    
    return {
        'X': X,
        'y': y,
        'measurement_ids': measurement_ids,
        'X_mask': X_mask,
        'y_mask': y_mask
    }


def make_longitudinal_data_loader(X_long, y_long, measurement_ids, n_timepoints, batch_size = 32):
    # Initialize datasets
    dataset = LongitudinalDataset(X_long, y_long, measurement_ids, time_steps=n_timepoints)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    return loader


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

    def training_loop(self, model, optimizer, num_epochs):

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0

            # beta = c_annealer.get_beta(epoch) * model.beta

            for batch_data in self.train_dataloader:
                optimizer.zero_grad()
                
                model_output = model(batch_data)
                loss = model.loss(model_output, batch_data)
                
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
            
            self.losses["train"].append(train_loss / self.len_train)
            print(f'Epoch {epoch+1}, Loss: {train_loss / self.len_train:.4f}')
            
            if self.validation:
                val_loss = 0
                with torch.no_grad():
                    for batch_data in self.val_dataloader:
                        
                        model_output = model(batch_data)
                        loss = model.loss(model_output, batch_data)
                        
                        val_loss += loss.item()
                self.losses["val"].append(val_loss / self.len_val)
                print(f'Epoch {epoch+1}, Validation Loss: {val_loss / self.len_val:.4f}')


class KLAnnealer:
    def __init__(self, total_epochs, anneal_start=0.2, anneal_end=0.8):
        self.total_epochs = total_epochs
        self.anneal_start = anneal_start
        self.anneal_end = anneal_end
    
    def get_beta(self, epoch):
        """Returns annealing factor between 0 and 1"""
        if epoch < self.anneal_start * self.total_epochs:
            return 0.0
        elif epoch > self.anneal_end * self.total_epochs:
            return 1.0
        else:
            return (epoch - self.anneal_start * self.total_epochs) / \
                   ((self.anneal_end - self.anneal_start) * self.total_epochs)


class CyclicAnnealer:
    def __init__(self, cycle_length, min_beta=0.0, max_beta=1.0, mode='cosine'):
        self.cycle_length = cycle_length
        self.min_beta = min_beta
        self.max_beta = max_beta
        self.mode = mode  # 'cosine' or 'linear'

    def get_beta(self, epoch):
        cycle_pos = epoch % self.cycle_length
        ratio = cycle_pos / self.cycle_length
        
        if self.mode == 'cosine':
            # Smooth cosine wave
            beta = self.min_beta + 0.5 * (self.max_beta - self.min_beta) * (1 - np.cos(np.pi * ratio))
        else:
            beta = self.max_beta - 1 * (ratio - 0.5) * (self.max_beta - self.min_beta)
        
        return beta


def print_correlations(X, X_hat):
    for i in range(X.shape[1]):
        corr, _ = pearsonr(X[:, i], X_hat[:, i])
        print(f"Dimension {i+1}: r = {corr:.3f}")


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
