import pandas as pd
import numpy as np
from pathlib import Path
from .splitter import split
from .scaler import scale
from .window_torch import torch_windowing, dataset_to_numpy_array

class TSDataset():

    def __init__(self, parameters) -> None:
        self.parameters = parameters

        self.dataset = self.parameters['dataset']['name']
        self.model_type = self.parameters['model']['params']['type']
        parent_directory = Path(__file__).resolve().parents[2]
        dataset_path = f'{parent_directory}/data/processed/{self.dataset}/data.csv'

        self.data = pd.read_csv(dataset_path)
        self.original_columns = self.data.drop(['year', 'tsgroup', 'time_idx'], axis=1, errors='ignore').columns
        self.data_processed = self.data.copy()
        self.feature_names = self.get_feature_names(self.data)
        self.target_names = self.get_target_names(self.data)
        self.label_idxs, self.values_idxs = self.get_values_and_labels_index(self.data)
        
        self.is_splitted = False
        self.is_windowed = False

        self.scaler = None

        self.data_train, self.data_valid, self.data_test = None, None, None



    def get_values_and_labels_index(self, data: pd.DataFrame) -> tuple:
        """
        Get the indices of value and label columns in the input data.

        Args:
            data (np.ndarray): Input data.

        Returns:
            tuple: Indices of label columns and value columns.
        """
        label_idxs = [idx for idx, col in enumerate(
            data.drop(['year', 'tsgroup', 'time_idx'], axis=1, errors='ignore').columns) if 'target' in col]
        values_idxs = [idx for idx, col in enumerate(
            data.drop(['year', 'tsgroup', 'time_idx'], axis=1, errors='ignore').columns) if 'target' not in col]

        return label_idxs, values_idxs
    
    def get_feature_names(self, data: pd.DataFrame):

        seq_len = self.parameters['dataset']['params']['seq_len']

        feature_names = np.array(
            [col for col in data.drop(['year', 'tsgroup', 'time_idx'], axis=1, errors='ignore').columns])

        features = np.array([np.core.defchararray.add(
            feature_names, ' t-'+str(i)) for i in range(seq_len, 0, -1)]).flatten()

        return features

    def get_target_names(self, data: pd.DataFrame):

        pred_len = self.parameters['dataset']['params']['pred_len']

        target_names = np.array(
            [col for col in data.drop(['year', 'tsgroup', 'time_idx'], axis=1, errors='ignore').columns if 'target' in col])

        features = np.array([np.core.defchararray.add(
            target_names, ' t+'+str(i)) for i in range(1, pred_len+1)]).flatten()

        return features
    
    def crop(self, start_year=0, end_year=9999):

        self.data_processed = self.data[(self.data.year>=start_year) & (self.data.year<=end_year)]

        return self

    def split(self):
        
        self.data_train, self.data_valid, self.data_test = split(self.data_processed, self.parameters)
        self.is_splitted = True

    def scale(self):
        
        self.data_train, self.data_valid, self.data_test, self.scaler = scale(self.data_train, self.data_valid, self.data_test)

        return self
    
    def numpy_to_pandas(self, data: np.array) -> pd.DataFrame:

        return pd.DataFrame(data=data, columns=self.original_columns)
    
    def windowing(self):
        
        if self.model_type=="pytorch":
            self.data_train, self.data_valid, self.data_test = torch_windowing(self.numpy_to_pandas(self.data_train), self.numpy_to_pandas(self.data_valid), self.numpy_to_pandas(self.data_test), values_idxs=self.values_idxs, label_idxs=self.label_idxs, parameters=self.parameters)            
        else: # In order to ensure the same windowing, torch is transformed to numpy
            self.data_train, self.data_valid, self.data_test = torch_windowing(self.numpy_to_pandas(self.data_train), self.numpy_to_pandas(self.data_valid), self.numpy_to_pandas(self.data_test), values_idxs=self.values_idxs, label_idxs=self.label_idxs, parameters=self.parameters)
            self.data_train = dataset_to_numpy_array(self.data_train)
            self.data_valid = dataset_to_numpy_array(self.data_valid)
            self.data_test = dataset_to_numpy_array(self.data_test)

        self.is_windowed = True
        return self

    def preprocess(self):

        self.split()
        self.scale()
        self.windowing()

        return self