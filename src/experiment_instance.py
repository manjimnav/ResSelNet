import numpy as np
import pandas as pd
from .model import get_model, get_selected_idxs
import tensorflow as tf
import time
import hashlib
import json
from datetime import datetime, timedelta
from typing import Tuple, Union, Optional
from sklearn.base import BaseEstimator
from .metric import MetricCalculator
from .dataset import TSDataset
import copy



class ExperimentInstance:

    """
    Initialize an ExperimentInstance which runs a single experiment with a set of parameters.

    Args:
        parameters (dict): Experiment parameters.
    """

    def __init__(self, parameters) -> None:
        
        self.parameters = parameters
        self.metrics = pd.DataFrame()
        self.dataset = None
        self.scaler = None
        self.model = None
        self.label_idxs, self.values_idxs = [], []
        self.code = self.dict_hash(parameters)

        parameters["code"] = self.code
        
        self.selected_idxs = []
        self.raw_results_ = []

    def convert(self, num):
        if isinstance(num, np.int64) or isinstance(num, np.int32): return int(num)  
        raise TypeError

    def dict_hash(self, dictionary:dict) -> str:
        """
        MD5 hash of the parameters used as experiment identifier.
        
        """
        dhash = hashlib.md5()
        # We need to sort arguments so {'a': 1, 'b': 2} is
        # the same as {'b': 2, 'a': 1}
        encoded = json.dumps(dictionary, sort_keys=True, default=self.convert).encode()
        dhash.update(encoded)
        return dhash.hexdigest()

    def train_tf(self, model: tf.keras.Model) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
        """
        Train a TensorFlow model.

        Args:
            model (tf.keras.Model): The model to train.
            data_train (tf.data.Dataset): Training data.
            data_valid (tf.data.Dataset): Validation data.

        Returns:
            Tuple[tf.keras.Model, tf.keras.callbacks.History]: Trained model and training history.
        """

        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

        history = model.fit(
            self.dataset.data_train["data"],
            epochs=100,
            callbacks=[callback],
            validation_data=self.dataset.data_valid["data"],
            verbose = 0
        )

        if 'TimeSelectionLayer' in self.parameters['selection']['name']:
            self.selected_idxs = get_selected_idxs(model, self.dataset.feature_names)

        return model, history
    
    def train_sk(self, model: BaseEstimator) -> Tuple[BaseEstimator, None]:
        """
        Train a scikit-learn model.

        Args:
            model: The scikit-learn model to train.
            data_train (tuple): Training data as a tuple of inputs and outputs.

        Returns:
            Tuple[Any, None]: Trained model and None (no training history).
        """

        data_train = self.dataset.data_train["data"]

        model_name = self.parameters['model']['name']

        model.fit(data_train[0], data_train[1])

        features = self.dataset.feature_names
        features_idxs = np.arange(0, features.flatten().shape[0])

        if model_name == 'lasso':
            importances = model.coef_.max(axis=0)
        else:
            importances = model.feature_importances_
        self.selected_idxs = {"level0": features_idxs[importances>0]}

        return model, None

    def train(self, model: Union[tf.keras.Model, BaseEstimator]) -> Tuple[Union[tf.keras.Model, BaseEstimator], Optional[tf.keras.callbacks.History]]:
        """
        Train a model.

        Args:
            model: The model to train.
            data_train: Training data.
            data_valid: Validation data.

        Returns:
            Tuple[Union[tf.keras.Model, Any], Optional[tf.keras.callbacks.History]]: Trained model and training history (if available).
        """

        model_type = self.parameters['model']['params']['type']

        if model_type == 'tensorflow':
            model, history = self.train_tf(model)
        else:
            model, history = self.train_sk(model)

        return model, history
    
    def execute_one(self) -> Tuple[pd.DataFrame, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Execute a single experiment instance.

        Returns:
            Tuple[pd.DataFrame, tf.Tensor, tf.Tensor, tf.Tensor]: Metrics DataFrame, test data inputs, true values, and predictions.
        """

        self.dataset.preprocess()

        n_features_in = len(self.dataset.label_idxs) + len(self.dataset.values_idxs)
        n_features_out = len(self.dataset.label_idxs)

        model = get_model(copy.deepcopy(self.parameters), n_features_in, n_features_out) # Important the deepcopy to avoid tf redefinition of parameters 

        start = time.time()
        self.model, history = self.train(model)
        duration = time.time() - start

        metric_calculator = MetricCalculator(self.dataset, self.parameters, selected_idxs = self.selected_idxs)

        metrics = metric_calculator.export_metrics(self.model, history, duration)

        return metric_calculator, metrics
    
    def store_raw_results(self, metric_calculator, test_year=None):
        if "year" in self.dataset.data.columns:
            dates = pd.date_range(datetime(self.dataset.data["year"].max(), 1, 1) + timedelta(hours=self.parameters['dataset']['params']['seq_len']), datetime(test_year, 12, 31), freq='H')
            dates = dates[:len(metric_calculator.true_test)]
            self.raw_results_.append((dates, metric_calculator.inputs_test, metric_calculator.true_test, metric_calculator.predictions_test))
        else:
            self.raw_results_.append((metric_calculator.inputs_test, metric_calculator.true_test, metric_calculator.predictions_test))
    
    def run(self) -> pd.DataFrame:
        """
        Run the experiment instance.

        Returns:
            pd.DataFrame: Metrics DataFrame.
        """
        
        self.dataset = TSDataset(self.parameters)

        split_by_year = self.parameters['dataset']['params'].get('crossval', False)

        self.metrics = pd.DataFrame()
        if split_by_year:

            years = sorted(self.dataset.data.year.unique())

            years = years[7:] # At least 7 years: 1 Test, 1 Val, 5 Train
            for test_year in years: # yearly crossval
                self.parameters['dataset']['params']['test_year'] = test_year

                self.dataset.crop(start_year=test_year-6, end_year=test_year)

                metric_calculator, year_metrics = self.execute_one()

                self.metrics = pd.concat([self.metrics, year_metrics])

                self.store_raw_results(metric_calculator, test_year=test_year)
        else:
            if "year" in self.dataset.data.columns:
                self.parameters['dataset']['params']['test_year'] = self.dataset.data["year"].max()
                
            metric_calculator, self.metrics = self.execute_one() 

            self.store_raw_results(metric_calculator, test_year=self.parameters['dataset']['params'].get('test_year', None))

        return self.metrics
