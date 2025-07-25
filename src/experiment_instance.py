import numpy as np
import pandas as pd
from .model import get_model
from torch.nn import Module
import time
import hashlib
import json
from datetime import datetime, timedelta
from typing import Tuple, Union
from sklearn.base import BaseEstimator
from .metric import MetricCalculator
from .dataset import TSDataset
import copy
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
import yaml
import wandb


def pformat(dictionary, function):
    if isinstance(dictionary, dict):
        return type(dictionary)((key, pformat(value, function)) for key, value in dictionary.items())
    if isinstance(dictionary, list):
        return type(dictionary)(pformat(value, function) for value in dictionary)
    if isinstance(dictionary, float):
        return function(dictionary)
    return dictionary

class ExperimentInstance:

    """
    Initialize an ExperimentInstance which runs a single experiment with a set of parameters.

    Args:
        parameters (dict): Experiment parameters.
    """

    def __init__(self, parameters, config_name) -> None:
        
        self.parameters = parameters
        self.config_name = config_name
        self.metrics = pd.DataFrame()
        self.dataset = None
        self.scaler = None
        self.model = None
        self.label_idxs, self.values_idxs = [], []


        self.code = self.dict_hash(self.parameters)
        self.parameters["code"] = self.code

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
        encoded = json.dumps(pformat(dictionary, lambda value: round(value, 12)), sort_keys=True, default=self.convert).encode()
        dhash.update(encoded)
        return dhash.hexdigest()

    def train_torch(self, model: Module) -> Tuple[Module, None]:
        """
        Train a PyTorch model.

        Args:
            model (tf.keras.Model): The model to train.
            data_train (tf.data.Dataset): Training data.
            data_valid (tf.data.Dataset): Validation data.

        Returns:
            Tuple[tf.keras.Model, tf.keras.callbacks.History]: Trained model and training history.
        """

        early_stop_callback = EarlyStopping(monitor="val/loss", min_delta=0.00, patience=10, verbose=False, mode="min")
        checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints/{self.config_name}/", filename=f"{self.code}-{self.parameters['dataset']['params'].get('test_year', '0')}", save_top_k=1, monitor="val/loss", mode="min")

        trainer = Trainer(max_epochs=100, callbacks=[early_stop_callback, checkpoint_callback], log_every_n_steps=8, enable_model_summary=False, enable_progress_bar=True)

        trainer.fit(
            model, 
            train_dataloaders=self.dataset.data_train["data"],
            val_dataloaders=self.dataset.data_valid["data"]
        )

        if checkpoint_callback.best_model_path != "":
            best_model = model.__class__.load_from_checkpoint(checkpoint_callback.best_model_path, model=model.model)
            best_model.eval().to('cpu')
        else:
            best_model = model
            best_model.eval().to('cpu')
        
        return best_model
    
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

        n_instances = data_train[0].shape[0]
        model.fit(data_train[0].reshape(n_instances, -1), data_train[1].reshape(n_instances, -1))

        features = self.dataset.feature_names
        features_idxs = np.arange(0, features.flatten().shape[0])

        if model_name == 'lasso':
            importances = model.coef_.max(axis=0)
        elif model_name == 'decisiontree':
            importances = model.feature_importances_
        else:
            importances = np.ones(len(features_idxs))

        self.selected_idxs = {"level0": features_idxs[importances>0].tolist()}

        return model

    def train(self, model: Union[Module, BaseEstimator]) -> Tuple[Union[Module, BaseEstimator]]:
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

        if model_type == 'pytorch':
            model = self.train_torch(model)
        else:
            model = self.train_sk(model)

        return model
    
    def execute_one(self) -> Tuple[pd.DataFrame, Tuple]:
        """
        Execute a single experiment instance.

        Returns:
            Tuple[pd.DataFrame, Tuple]: Metrics DataFrame, test data inputs, true values, and predictions.
        """

        #self.wandb_logger = WandbLogger(name=f"{self.code}-{self.parameters['dataset']['params'].get('test_year', '0')}", project=self.parameters['dataset']["name"], offline=True)

        #self.wandb_logger.experiment.config.update(self.parameters)
        
        self.dataset.preprocess()

        n_features_in = len(self.dataset.label_idxs) + len(self.dataset.values_idxs)
        n_features_out = len(self.dataset.label_idxs)

        model = get_model(copy.deepcopy(self.parameters), n_features_in, n_features_out) # Important the deepcopy to avoid tf redefinition of parameters 

        start = time.time()
        self.model = self.train(model)
        duration = time.time() - start

        metric_calculator = MetricCalculator(self.dataset, self.parameters, selected_idxs = self.selected_idxs)

        metrics = metric_calculator.export_metrics(self.model, duration)

        #self.wandb_logger.log_metrics(metrics.to_dict())

        #self.wandb_logger.log_table(key="metrics", dataframe=metrics)

        wandb.finish()

        return metric_calculator, metrics
    
    def store_raw_results(self, metric_calculator, test_year=None):
        if "year" in self.dataset.data.columns:
            dates = pd.date_range(datetime(self.dataset.data["year"].max(), 1, 1) + timedelta(hours=self.parameters['dataset']['params']['seq_len']), datetime(test_year, 12, 31), freq='h')
            dates = dates[:len(metric_calculator.true_test)]
            self.raw_results_.append((test_year, dates, metric_calculator.inputs_test, metric_calculator.true_test, metric_calculator.predictions_test))
        else:
            self.raw_results_.append((metric_calculator.inputs_test, metric_calculator.true_test, metric_calculator.predictions_test))

        """with open(f"configuration/{self.parameters['dataset']['name']}/experiment_hyperparameters/{self.code}.yml", 'w') as outfile:
            yaml.dump(self.parameters, outfile, default_flow_style=False)"""

    def run(self) -> pd.DataFrame:
        """
        Run the experiment instance.

        Returns:
            pd.DataFrame: Metrics DataFrame.
        """
        
        self.dataset = TSDataset(self.parameters)

        self.metrics = pd.DataFrame()
        metric_calculator, self.metrics = self.execute_one() 
        self.store_raw_results(metric_calculator, test_year=self.parameters['dataset']['params'].get('test_year', None))

        return self.metrics
