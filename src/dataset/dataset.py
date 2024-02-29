import pandas as pd
import numpy as np
from pathlib import Path
from .splitter import split
from .scaler import scale
from .window import windowing

class TSDataset():

    def __init__(self, parameters) -> None:
        self.parameters = parameters

        self.dataset = self.parameters['dataset']['name']
        parent_directory = Path(__file__).resolve().parents[2]
        dataset_path = f'{parent_directory}/data/processed/{self.dataset}/data.csv'

        self.data = pd.read_csv(dataset_path)
        self.data_processed = self.data.copy()
        self.feature_names = self.get_feature_names(self.data)
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
            data.drop(['year', 'tsgroup'], axis=1, errors='ignore').columns) if 'target' in col]
        values_idxs = [idx for idx, col in enumerate(
            data.drop(['year', 'tsgroup'], axis=1, errors='ignore').columns) if 'target' not in col]

        return label_idxs, values_idxs
    
    def get_feature_names(self, data: pd.DataFrame):

        seq_len = self.parameters['dataset']['params']['seq_len']

        feature_names = np.array(
            [col for col in data.drop(['year', 'tsgroup'], axis=1, errors='ignore').columns])

        features = np.array([np.core.defchararray.add(
            feature_names, ' t-'+str(i)) for i in range(seq_len, 0, -1)]).flatten()

        return features
    
    def crop(self, start_year=0, end_year=9999):

        self.data_processed = self.data[(self.data.year>=start_year) & (self.data.year<=end_year)]

        return self

    def split(self):
        
        self.data_train, self.data_valid, self.data_test = split(self.data_processed, self.parameters)
        self.is_splitted = True

        return self


    def scale(self):
        
        self.data_train, self.data_valid, self.data_test, self.scaler = scale(self.data_train, self.data_valid, self.data_test)

        return self
    
    def windowing(self):
        self.is_windowed = True
        self.data_train, self.data_valid, self.data_test = windowing(self.data_train, self.data_valid, self.data_test, self.values_idxs, self.label_idxs, self.parameters)

        return self

    def preprocess(self):

        self.split()
        self.scale()
        self.windowing()

        return self