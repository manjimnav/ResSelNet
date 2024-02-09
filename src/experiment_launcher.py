import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
import random
import os
from copy import deepcopy
from bayes_opt import BayesianOptimization, UtilityFunction
from typing import Iterable, Union
from tqdm import tqdm
from tqdm.contrib import itertools
from .experiment_instance import ExperimentInstance

class ExperimentLauncher:

    """
    Initialize an ExperimentLauncher.

    Args:
        config_path (str): Path to the configuration files.
        save_file (str, optional): Path to save the results CSV file. Defaults to "../results/TimeSelection/results.csv".
        search_type (str, optional): Search type, either 'grid' or 'bayesian'. Defaults to 'grid'.
        iterations (int, optional): Number of iterations for Bayesian optimization. Defaults to 10.
    """

    def __init__(self, config_path, save_file="../results/TimeSelection/results.csv", search_type='grid', iterations=10) -> None:
        
        data_config, selection_config, model_config = f"{config_path}data_config.yaml", f"{config_path}selection_config.yaml", f"{config_path}model_config.yaml"

        self.data_configuration = self.load_config(data_config)
        self.selection_configuration = self.load_config(selection_config)
        self.model_configuration = self.load_config(model_config)
        self.save_file = save_file
        self.search_type = search_type
        self.iterations = iterations
        self.optimizer = None


        if os.path.exists(self.save_file):
            self.metrics = pd.read_csv(self.save_file)
        else:
            self.metrics = pd.DataFrame()
    
    def nested_product(self, configurations: Union[dict, list]) -> Iterable:
        """
        Generate nested product configurations.

        Args:
            configurations: Configuration dictionary.

        Yields:
            Iterable: Nested product configurations.
        """
        if isinstance(configurations, list):
            for value in configurations:
                yield from ([value] if not isinstance(value, (dict, list)) else self.nested_product(value))
        elif isinstance(configurations, dict):
            for key, value in configurations.items():
                if isinstance(value, list) and len(value)==3 and isinstance(value[2], dict) and 'step' in value[2]:
                    configurations[key] = list(np.arange(value[0], value[1]+value[2]['step'], value[2]['step']))
                    
            for i in itertools.product(*map(self.nested_product, configurations.values())):
                yield dict(zip(configurations.keys(), i))
        else:
            yield configurations

    def load_config(self, path: str) -> dict:
        """
        Load a YAML configuration file.

        Args:
            path (str): Path to the YAML configuration file.

        Returns:
            dict: Loaded configuration as a dictionary.
        """
        config = {}
        with open(path, "r") as stream:
            try:
                config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                raise Exception("Configuration at path {path} was not found")
        return config
    
    def seed(self, seed: int = 123) -> None:
        """
        Seed random number generators for reproducibility.

        Args:
            seed (int, optional): Random seed. Defaults to 123.
        """
        tf.random.set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    def transform_to_bounds(self, general_params: dict) -> dict:
        """
        Transform general parameters to bounds for Bayesian optimization.

        Args:
            general_params (dict): General parameters.

        Returns:
            dict: Bounds for Bayesian optimization.
        """
        bounds = {}
        for general_key in general_params.keys():
            if general_params[general_key]['params'] is not None:
                bound_params = {key: value for key, value in general_params[general_key]['params'].items() if type(value) == list}

                bounds.update(bound_params)
        
        return bounds
    
    def update_params(self, optimized_params: dict, general_params: dict) -> dict:
        """
        Update general parameters with optimized values.

        Args:
            optimized_params (dict): Optimized parameters.
            general_params (dict): General parameters.

        Returns:
            dict: Updated general parameters.
        """
        
        params = deepcopy(general_params)
        for general_key in params.keys():
            if params[general_key]['params'] is not None:
                keys_to_update = list(params[general_key]['params'].keys())
                for key_to_update in keys_to_update:
                    
                    if key_to_update in optimized_params:
                        BuiltinClass = params[general_key]['params'][key_to_update][0].__class__
                        params[general_key]['params'][key_to_update] = BuiltinClass(optimized_params[key_to_update])
        return params
    
    def bayesian_optimization(self, general_params: dict) -> Iterable:
        """
        Perform Bayesian optimization for hyperparameter search.

        Args:
            general_params (dict): General parameters.

        Yields:
            Iterable: Hyperparameter configurations.
        """

        bounds = self.transform_to_bounds(general_params)
        self.optimizer = BayesianOptimization(
            f=None,
            pbounds=bounds,
            verbose=2,
            random_state=1,
        )

        utility = UtilityFunction(kind="ucb", kappa=2.576)

        for _ in range(self.iterations):
            optimized_params = self.optimizer.suggest(utility)      

            params = self.update_params(optimized_params, general_params)

            yield params

    def search_hyperparameters(self, general_params: dict) -> Iterable:
        """
        Search hyperparameters using grid search or Bayesian optimization.

        Args:
            general_params (dict): General parameters.

        Yields:
            Iterable: Hyperparameter configurations.
        """
        if self.search_type == 'grid':
            yield from self.nested_product(general_params)
        elif self.search_type == 'bayesian':
            yield from self.bayesian_optimization(general_params)
    
    def is_performed(self, experiment, params):

        if experiment.code in self.metrics.get('code', default=pd.Series([], dtype=str)).tolist():
            if self.optimizer == 'bayesian': # Register previous metrics performed
                self.optimizer.register(params=params, target=-self.metrics.loc[self.metrics.code == experiment.code,'root_mean_squared_error_valid'].mean())
                return True
        
        return False

    def run(self) -> pd.DataFrame:
        """
        Run the experiment launcher.

        Returns:
            pd.DataFrame: Metrics DataFrame.
        """

        for dataset, selection, model in tqdm(itertools.product(self.data_configuration.keys(), self.selection_configuration.keys(), self.model_configuration.keys())):
            
            dataset_params = {"dataset": {'name': dataset, "params": self.data_configuration[dataset]}}
            selection_params = {"selection": {"name": selection, "params": self.selection_configuration[selection]}}
            model_params = {"model": {"name": model, "params": self.model_configuration[model]}}

            general_params = {**dataset_params, **selection_params, **model_params}

            for params in tqdm(self.search_hyperparameters(general_params), leave=False, total=self.iterations):

                if (params['model']['params']['type'] == "sklearn" and params['selection']['name'] != 'NoSelection'):
                    continue   
                
                self.seed()                
                experiment = ExperimentInstance(params)
                
                if self.is_performed(experiment, params):
                    continue

                metrics = experiment.run()

                if self.optimizer == 'bayesian': self.optimizer.register(params=params, target=-metrics['root_mean_squared_error_valid'].mean())
            
                self.metrics = pd.concat([self.metrics, metrics])

                self.metrics.to_csv(self.save_file, index=None)
        
        return self.metrics