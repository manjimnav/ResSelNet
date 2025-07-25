import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from typing import Union

class TimeSeriesDataset(Dataset):
    def __init__(self, df, target_cols, seq_length, pred_length, shift):
        """
        Args:
            df (pd.DataFrame): Input DataFrame containing the time series data.
            target_cols (list): List of target column names.
            seq_length (int): Length of the input sequence.
            pred_length (int): Length of the prediction sequence.
            shift (int): Shift between windows.
        """
        self.df = df
        self.target_cols = target_cols
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.shift = shift

        self.features = df.values
        self.targets = df[target_cols].values
        self.samples = self._create_samples()

    def _create_samples(self):
        num_samples = (len(self.df) - self.seq_length - self.pred_length) // self.shift + 1
        samples = [(i * self.shift, i * self.shift + self.seq_length, i * self.shift + self.seq_length + self.pred_length) for i in range(num_samples)]
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        start_idx, mid_idx, end_idx = self.samples[idx]
        
        seq_x = self.features[start_idx:mid_idx]
        seq_y = self.targets[mid_idx:end_idx]
        
        return torch.tensor(seq_x, dtype=torch.double), torch.tensor(seq_y, dtype=torch.double)

def window_one(data: pd.DataFrame, values_idxs: list, label_idxs: list, parameters: dict, shuffle=False):

    target = data.columns[label_idxs].tolist()
    batch_size = parameters['model']['params'].get('batch_size', 32)
    seq_len = parameters['dataset']['params']['seq_len']
    pred_len = parameters['dataset']['params']['pred_len']
    shift = parameters['dataset']['params']['shift'] or seq_len

    data_windowed = DataLoader(TimeSeriesDataset(
                data,
                target_cols=target,
                seq_length=seq_len,
                pred_length=pred_len,
                shift=shift
            ), batch_size=batch_size, shuffle=shuffle)
    
    return {"data": data_windowed }

def window_grouped(data: pd.DataFrame, values_idxs: list, label_idxs: list, parameters: dict, shuffle=False):

    data_windowed = []
    groups = []
    for tsgroup, group_values in data:
        group_windowed = window_one(group_values, values_idxs, label_idxs, parameters, shuffle=shuffle)
        data_windowed.append(group_windowed)
        groups.extend([tsgroup for _ in range(len(group_windowed))])
    
    data_windowed = torch.utils.data.ConcatDataset(data_windowed)

    return {"data": data_windowed, "groups": groups}

def torch_windowing(train_scaled: Union[pd.DataFrame, tuple], valid_scaled: Union[pd.DataFrame, tuple], test_scaled: Union[pd.DataFrame, tuple], values_idxs: list, label_idxs: list, parameters: dict):

    if type(train_scaled)==tuple:
        data_train = window_grouped(train_scaled, values_idxs, label_idxs, parameters, shuffle=True)
        data_valid = window_grouped(valid_scaled, values_idxs, label_idxs, parameters, shuffle=False)
        data_test = window_grouped(test_scaled, values_idxs, label_idxs, parameters, shuffle=False)
    else:
        data_train = window_one(train_scaled, values_idxs, label_idxs, parameters, shuffle=True)
        data_valid = window_one(valid_scaled, values_idxs, label_idxs, parameters, shuffle=False)
        data_test = window_one(test_scaled, values_idxs, label_idxs, parameters, shuffle=False)
    
    return data_train, data_valid, data_test


def dataset_to_numpy_array(dataset):

    X_values = []
    y_values = []
    for x, y in dataset["data"]:

        X_values.append(x.cpu().detach().numpy())
        y_values.append(y.cpu().detach().numpy())
    
    X_values = np.concatenate(X_values)
    y_values = np.concatenate(y_values)
    numpy_array = [X_values, y_values]

    result_array = {"data": numpy_array}

    if "groups" in dataset:
        result_array["groups"] = dataset["groups"]

    return result_array
        
        

    