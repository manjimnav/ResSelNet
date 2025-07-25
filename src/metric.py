import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from typing import Tuple, Union, Iterable
from sklearn.base import BaseEstimator
from .dataset import TSDataset, dataset_to_numpy_array, inverse_scale
from typing import Dict, Union
import torch

class MetricCalculator():

    def __init__(self, dataset: TSDataset, parameters: dict, selected_idxs:  Dict[str, list], metrics_names=['mae', 'mse', 'rmse', 'r2', 'mape', 'mase', 'wape']) -> None:
        
        self.dataset = parameters["dataset"]["name"]
        self.dataset = dataset
        self.parameters = parameters
        self.label_idxs = dataset.label_idxs
        self.features_names = dataset.feature_names
        self.metrics_names = metrics_names
        self.selected_idxs = selected_idxs

        self.predictions_test, self.true_test, self.inputs_test, self.inputs_valid = None, None, None, None

        root_mean_squared_error = lambda true, predictions: np.sqrt(mean_squared_error(true, predictions))

        mase = lambda true, predictions: np.mean([abs(true[i] - predictions[i]) / (abs(true[i] - true[i - 1]) / len(true) - 1) for i in range(1, len(true))])
        wape = lambda true, predictions: np.abs(true - predictions).sum() / true.sum()

        self.name_to_metric = {
            'mae': mean_absolute_error,
            'mse': mean_squared_error,
            'rmse': root_mean_squared_error,
            'mape': mean_absolute_percentage_error,
            'mase': mase,
            'wape': wape,
            'r2': r2_score
        }

        
    def recursive_items(self, dictionary, parent_key=None) -> Iterable:
        """
        Recursively iterate over dictionary items.

        Args:
            dictionary (dict): The input dictionary.
            parent_key (str, optional): The parent key in the recursion. Defaults to None.

        Yields:
            Iterable: Key-value pairs.
        """
        for key, value in dictionary.items():
            key = key if parent_key is None else parent_key+'_'+key
            if type(value) is dict:
                yield from self.recursive_items(value, parent_key=key)
            else:
                yield (key, value)

    def calculate_metrics(self, true, predictions, metric_key=''):

        return {metric_name+metric_key: self.name_to_metric[metric_name](true, predictions) for metric_name in self.metrics_names}
    
    def include_metadata(self, metrics, duration):

        for key, value in self.recursive_items(self.parameters):
            metrics[key] = value
        
        if type(self.selected_idxs) == list:
            metrics['selected_features'] = [self.selected_idxs]
        else:
            metrics['selected_features'] = str(self.selected_idxs)

        metrics['duration'] = duration

        return metrics

    def export_metrics(self, model: Union[torch.nn.Module, BaseEstimator], duration: float) -> Tuple[pd.DataFrame, Tuple]:
        
        if self.dataset.model_type != "pytorch":
            self.inputs_test = self.dataset.data_test["data"][0]
            self.inputs_valid = self.dataset.data_valid["data"][0]
            
            n_instances_test = self.dataset.data_test["data"][1].shape[0]
            n_instances_valid = self.dataset.data_valid["data"][1].shape[0]
            true_scaled, true_valid_scaled = self.dataset.data_test["data"][1].reshape(n_instances_test, -1), self.dataset.data_valid["data"][1].reshape(n_instances_valid, -1)
        else:
            data_test_numpy = dataset_to_numpy_array(self.dataset.data_test)["data"] if not isinstance(self.dataset.data_test, np.ndarray) else self.dataset.data_test
            data_valid_numpy = dataset_to_numpy_array(self.dataset.data_valid)["data"] if not isinstance(self.dataset.data_valid, np.ndarray) else self.dataset.data_valid
            self.inputs_test = data_test_numpy[0]
            self.inputs_valid = data_valid_numpy[0]
            true_scaled, true_valid_scaled = data_test_numpy[1], data_valid_numpy[1]

        if self.dataset.model_type != "pytorch":
            predictions = model.predict(self.inputs_test.reshape(self.inputs_test.shape[0], -1))
            predictions_valid = model.predict(self.inputs_valid.reshape(self.inputs_valid.shape[0], -1))
        else:
            predictions = model(torch.tensor(self.inputs_test), torch.tensor(true_scaled)).cpu().detach().numpy()
            predictions_valid = model(torch.tensor(self.inputs_valid), torch.tensor(true_valid_scaled)).cpu().detach().numpy()

        predictions_scaled, predictions_valid_scaled  = predictions, predictions_valid
        true, predictions = inverse_scale([true_scaled, predictions_scaled], groups=self.dataset.data_test.get("groups", None), scaler=self.dataset.scaler, label_idxs=self.dataset.label_idxs)
        true_valid, predictions_valid = inverse_scale([true_valid_scaled, predictions_valid_scaled], groups=self.dataset.data_valid.get("groups", None), scaler=self.dataset.scaler, label_idxs=self.dataset.label_idxs)

        metrics_test = self.calculate_metrics(true, predictions)
        metrics_valid = self.calculate_metrics(true_valid, predictions_valid, metric_key='_valid')

        metrics = pd.DataFrame({**metrics_test, **metrics_valid}, index=[0])

        metrics = self.include_metadata(metrics, duration)

        self.true_test = true
        self.predictions_test = predictions
        
        return metrics