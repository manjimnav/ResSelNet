import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
import tensorflow as tf
from typing import Tuple, Union, Iterable
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

class MetricCalculator():

    def __init__(self, scaler: TransformerMixin, parameters: dict, label_idxs: np.array, features_names: pd.Index, metrics_names=['mae', 'mse', 'rmse', 'r2', 'mape']) -> None:
        
        self.dataset = parameters["dataset"]["name"]
        self.scaler = scaler
        self.parameters = parameters
        self.label_idxs = label_idxs
        self.features_names = features_names
        self.metrics_names = metrics_names

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
    
    def inverse_scale(self, *y_vals):

        mean = self.scaler.mean_[self.label_idxs]
        std = self.scaler.scale_[self.label_idxs]

        iscaled_values = ()
        for y in y_vals:
            y_inverse = y*std + mean
            iscaled_values + (y_inverse,)
        
        return iscaled_values

    def calculate_metrics(self, true, predictions, metric_key=''):

        return {metric_name+metric_key: self.name_to_metric[metric_name](true, predictions) for metric_name in self.metrics_names}
    
    def include_metadata(self, metrics, history, duration):

        for key, value in self.recursive_items(self.parameters):
            metrics[key] = value
        
        metrics['selected_features'] = [self.features_names[list(self.selected_idxs)].tolist()]

        metrics['duration'] = duration

        if history is not None:
            metrics['history'] = str(history.history.get('val_loss', None))
            metrics['val_loss'] = min(history.history.get('val_loss', None))

        return metrics

    def export_metrics(self, model: Union[tf.keras.Model, BaseEstimator], history: tf.keras.callbacks.History, data_test: np.ndarray, data_valid: np.ndarray, duration: float) -> Tuple[pd.DataFrame, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Calculate evaluation metrics.

        Args:
            model: The trained model.
            history: The training history (optional).
            data_test (tuple): Test data as a tuple of inputs and outputs.
            data_valid (tf.data.Dataset): Validation data.
            duration (float): Duration of training.

        Returns:
            pd.DataFrame: Metrics DataFrame
        """
        
        self.inputs_test = data_test[0]
        self.inputs_valid = data_valid[0]

        predictions = model.predict(self.inputs_test)
        predictions_valid = model.predict(self.inputs_valid)

        true_scaled, true_valid_scaled, predictions_scaled, predictions_valid_scaled  = data_test[1], data_valid[1], predictions, predictions_valid
        true, true_valid, predictions, predictions_valid = self.inverse_scale(true_scaled, true_valid_scaled, predictions_scaled, predictions_valid_scaled, std=std, mean=mean)

        metrics_test = self.calculate_metrics(self, true, predictions)
        metrics_valid = self.calculate_metrics(self, true_valid, predictions_valid, metric_key='_valid')

        metrics = pd.DataFrame({**metrics_test, **metrics_valid}, index=[0])

        metrics = self.include_metadata(metrics, history, duration)

        self.true_test = true.flatten()
        self.predictions_test = predictions.flatten()
        
        return metrics