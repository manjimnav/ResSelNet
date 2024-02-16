import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
import tensorflow as tf
from typing import Tuple, Union, Iterable
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

class MetricCalculator():

    def __init__(self, scaler: TransformerMixin, parameters: dict, label_idxs: np.array, features_names: pd.Index, selected_idxs: list | dict[list], metrics_names=['mae', 'mse', 'rmse', 'r2', 'mape']) -> None:
        
        self.dataset = parameters["dataset"]["name"]
        self.scaler = scaler
        self.parameters = parameters
        self.label_idxs = label_idxs
        self.features_names = features_names
        self.metrics_names = metrics_names
        self.selected_idxs = selected_idxs

        self.predictions_test, self.true_test, self.inputs_test, self.inputs_valid = None, None, None, None

        root_mean_squared_error = lambda true, predictions: np.sqrt(mean_squared_error(true, predictions))

        self.name_to_metric = {
            'mae': mean_absolute_error,
            'mse': mean_squared_error,
            'rmse': root_mean_squared_error,
            'mape': mean_absolute_percentage_error,
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
    
    def inverse_scale(self, y_vals, groups=None):

        if groups is None:
            mean = self.scaler.mean_[self.label_idxs]
            std = self.scaler.scale_[self.label_idxs]

            iscaled_values = ()
            for y in y_vals:
                y_inverse = y*std + mean
                iscaled_values + (y_inverse,)
        else:
            iscaled_values = ()
            for y_scaled in y_vals:
                y_inverse = []
                for group, y_val in zip(groups, y_scaled):
                    mean = self.scaler.scalers[group].mean_[self.label_idxs]
                    std = self.scaler.scalers[group].scale_[self.label_idxs]
                    y_inverse.append(y_val*std + mean)

                iscaled_values + (y_inverse,)
        
        return iscaled_values

    def calculate_metrics(self, true, predictions, metric_key=''):

        return {metric_name+metric_key: self.name_to_metric[metric_name](true, predictions) for metric_name in self.metrics_names}
    
    def include_metadata(self, metrics, history, duration):

        for key, value in self.recursive_items(self.parameters):
            metrics[key] = value
        
        if type(self.selected_idxs) == list:
            metrics['selected_features'] = [self.features_names[self.selected_idxs].tolist()]
        else:

            selected_features = {}
            for layer_name, idxs in self.selected_idxs.items():
                selected_features[layer_name] = self.features_names[idxs].tolist()

            metrics['selected_features'] = str(selected_features)

        metrics['duration'] = duration

        if history is not None:
            metrics['history'] = str(history.history.get('val_loss', None))
            metrics['val_loss'] = min(history.history.get('val_loss', None))

        return metrics

    def export_metrics(self, model: Union[tf.keras.Model, BaseEstimator], history: tf.keras.callbacks.History, data_test: np.ndarray, data_valid: np.ndarray, duration: float) -> Tuple[pd.DataFrame, tf.Tensor, tf.Tensor, tf.Tensor]:
        
        self.inputs_test = data_test["data"][0]
        self.inputs_valid = data_valid["data"][0]

        predictions = model.predict(self.inputs_test)
        predictions_valid = model.predict(self.inputs_valid)

        true_scaled, true_valid_scaled, predictions_scaled, predictions_valid_scaled  = data_test["data"][1], data_valid["data"][1], predictions, predictions_valid
        true, predictions = self.inverse_scale([true_scaled, predictions_scaled], groups=data_test.get("groups", None))
        true_valid, predictions_valid = self.inverse_scale([true_valid_scaled, predictions_valid_scaled], groups=data_valid.get("groups", None))

        metrics_test = self.calculate_metrics(self, true, predictions)
        metrics_valid = self.calculate_metrics(self, true_valid, predictions_valid, metric_key='_valid')

        metrics = pd.DataFrame({**metrics_test, **metrics_valid}, index=[0])

        metrics = self.include_metadata(metrics, history, duration)

        self.true_test = true.flatten()
        self.predictions_test = predictions.flatten()
        
        return metrics