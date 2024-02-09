import numpy as np
import pandas as pd
from .dataset import get_values_and_labels_index, split, scale, windowing, get_feature_names
from .selection import select_features
from .model import get_model, get_selected_idxs
from pathlib import Path
import tensorflow as tf
import time
import hashlib
import json
from datetime import datetime, timedelta
from typing import Tuple, Union, Optional
from sklearn.base import BaseEstimator
from .metric import MetricCalculator


class ExperimentInstance:

    """
    Initialize an ExperimentInstance which runs a single experiment with a set of parameters.

    Args:
        parameters (dict): Experiment parameters.
    """

    def __init__(self, parameters) -> None:
        
        self.parameters = parameters
        self.metrics = pd.DataFrame()
        self.data = None
        self.scaler = None
        self.model = None
        self.dataset = self.parameters['dataset']['name']
        self.label_idxs, self.values_idxs = [], []
        self.code = self.dict_hash(parameters)

        parameters["code"] = self.code
        
        self.selected_idxs = []
        self.raw_results_ = []
        self.data_train = None

        parent_directory = Path(__file__).resolve().parents[1]
        self.dataset_path = f'{parent_directory}/data/processed/{self.dataset}/data.csv'

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
        print(dictionary)
        encoded = json.dumps(dictionary, sort_keys=True, default=self.convert).encode()
        dhash.update(encoded)
        return dhash.hexdigest()

    def preprocess_data(self) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Preprocess the data for training.

        Returns:
            Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]: Training, validation, and test datasets.
        """
        self.label_idxs, self.values_idxs = get_values_and_labels_index(self.data)
        if len(self.values_idxs)<1 and not self.parameters['dataset']['params'].get('select_timesteps', True):
            raise Exception(f"Cannot select features in dataset {self.parameters['dataset']['name']}")
            
        train_df, valid_df, test_df = split(self.data, self.parameters)

        train_scaled, valid_scaled, test_scaled, self.scaler = scale(train_df, valid_df, test_df)

        self.selected_idxs = select_features(train_scaled, self.parameters, self.label_idxs)

        data_train, data_valid, data_test = windowing(train_scaled, valid_scaled, test_scaled, self.values_idxs, self.label_idxs, self.selected_idxs, self.parameters)
        
        #self.data_train = np.concatenate(list(map(lambda x: x.numpy(), next(data_train.batch(9999999999).__iter__())))[0])
        
        return data_train, data_valid, data_test

    def read_data(self) -> None:
        """Read data from the dataset path."""
        self.data = pd.read_csv(self.dataset_path)

        self.parameters["features"] = [get_feature_names(self.data, self.parameters)]


    def train_tf(self, model: tf.keras.Model, data_train: tf.data.Dataset, data_valid: tf.data.Dataset) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
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
            data_train,
            epochs=100,
            callbacks=[callback],
            validation_data=data_valid,
            verbose = 0
        )

        if 'TimeSelectionLayer' in self.parameters['selection']['name']:
            self.selected_idxs = get_selected_idxs(model, get_feature_names(self.data, self.parameters))

        return model, history
    
    def train_sk(self, model: BaseEstimator, data_train: np.ndarray) -> Tuple[BaseEstimator, None]:
        """
        Train a scikit-learn model.

        Args:
            model: The scikit-learn model to train.
            data_train (tuple): Training data as a tuple of inputs and outputs.

        Returns:
            Tuple[Any, None]: Trained model and None (no training history).
        """

        model_name = self.parameters['model']['name']

        model.fit(data_train[0], data_train[1])

        features = get_feature_names(self.data, self.parameters)
        features_idxs = np.arange(0, features.flatten().shape[0])

        if model_name == 'lasso':
            importances = model.coef_.max(axis=0)
        else:
            importances = model.feature_importances_
        self.selected_idxs = features_idxs[importances>0]

        return model, None

    def train(self, model: Union[tf.keras.Model, BaseEstimator], data_train, data_valid) -> Tuple[Union[tf.keras.Model, BaseEstimator], Optional[tf.keras.callbacks.History]]:
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
            model, history = self.train_tf(model, data_train, data_valid)
        else:
            model, history = self.train_sk(model, data_train)

        return model, history
    
    def execute_one(self) -> Tuple[pd.DataFrame, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Execute a single experiment instance.

        Returns:
            Tuple[pd.DataFrame, tf.Tensor, tf.Tensor, tf.Tensor]: Metrics DataFrame, test data inputs, true values, and predictions.
        """
        data_train, data_valid, data_test = self.preprocess_data()

        model = get_model(self.parameters, self.label_idxs, self.values_idxs)

        start = time.time()
        self.model, history = self.train(model, data_train, data_valid)
        duration = time.time() - start

        metric_calculator = MetricCalculator(self.scaler, self.parameters, label_idxs=self.label_idxs, features_names=get_feature_names(self.data))

        metrics = metric_calculator.export_metrics(self.model, history, data_test, data_valid, duration)

        return metrics
    
    def store_raw_results(self, inputs, true, predictions, test_year=None):
        if "year" in self.data.columns:
            dates = pd.date_range(datetime(self.data["year"].max(), 1, 1) + timedelta(hours=self.parameters['dataset']['params']['seq_len']), datetime(test_year, 12, 31), freq='H')
            dates = dates[:len(true)]
            self.raw_results_.append((dates, inputs, true, predictions))
        else:
            self.raw_results_.append((inputs, true, predictions))
    
    def run(self) -> pd.DataFrame:
        """
        Run the experiment instance.

        Returns:
            pd.DataFrame: Metrics DataFrame.
        """
        
        self.read_data()

        split_by_year = self.parameters['dataset']['params'].get('crossval', False)

        self.metrics = pd.DataFrame()
        if split_by_year:

            years = sorted(self.data.year.unique())

            years = years[5:] # At least 6 years: 1 Test, 1 Val, 4 Train
            data_complete = self.data.copy()
            for test_year in sorted(self.data.year.unique()): # yearly crossval
                self.parameters['dataset']['params']['test_year'] = test_year

                self.data = data_complete[data_complete.year<=test_year] # Implements scalated data

                year_metrics, inputs, true, predictions = self.execute_one()

                self.metrics = pd.concat([self.metrics, year_metrics])

                self.store_raw_results(self, inputs, true, predictions, test_year=test_year)
        else:
            if "year" in self.data.columns:
                self.parameters['dataset']['params']['test_year'] = self.data["year"].max()
                
            self.metrics, inputs, true, predictions = self.execute_one() 

            self.store_raw_results(self, inputs, true, predictions, test_year=self.parameters['dataset']['params'].get('test_year', None))

        return self.metrics
