import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
from sklearn.model_selection import train_test_split


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


def make_long_array(x, feature_dimensions=-1):
    # join the first two dimensions, i.e., patient and measurement
    x_shape = x.shape
    x = x.reshape(-1, x_shape[feature_dimensions])

    return x


def sum_nan_on_given_dims(x, feature_dimensions):
    return torch.isnan(x).sum(dim=feature_dimensions) if x.dim() > 1 else torch.isnan(x)


class CustomDataset(Dataset):
    def __init__(self, *arrays, reshape=False, remove_missing=True, feature_dimensions=-1, device=torch.device("cpu")):
        """
        Args:
        """
        self.device = device

        self.original_shapes = [arr.shape for arr in arrays]

        if reshape:
            arrays = [make_long_array(arr, feature_dimensions) for arr in arrays]
        
        # check for missing data
        is_missing = False
        for arr in arrays:
            if torch.isnan(arr)[:, ...].any():
                is_missing = True

        if is_missing:
            missing_idx = [sum_nan_on_given_dims(arr, feature_dimensions) for arr in arrays]

        # collect all missing indeces
        all_missing_idx = torch.stack(missing_idx, axis=1).sum(axis=1) > 0
        # filter out missing
        if remove_missing:
            arrays = [arr[~all_missing_idx, ...] for arr in arrays]
        
        self.new_shapes = [arr.shape for arr in arrays]
        self.arrays = arrays

        print("Input Tensors Shapes: ", self.original_shapes)
        print("New Tensors Shapes: ", self.new_shapes)

    def __len__(self):
        return len(self.arrays[0])

    def __getitem__(self, idx):
        return tuple(array[idx] for array in self.arrays)


def make_data_loader(
    *arrays,
    batch_size = 32,
    feature_dimensions=-1,
    reshape=False,
    drop_missing=True,
    device=torch.device("cpu")
):
    # Initialize datasets

    dataset = CustomDataset(
        *arrays,
        reshape=reshape,
        remove_missing=drop_missing,
        feature_dimensions=feature_dimensions,
        device=device
    )

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    return loader

# arrays = tensor_data_train
# ddd = make_data_loader(*tensor_data_train, reshape=True, drop_missing=True)
# [dd.shape for dd in next(iter(ddd))]


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
