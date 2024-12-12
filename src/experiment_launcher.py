import os
import yaml
import numpy as np
import pandas as pd
import random
from copy import deepcopy
from bayes_opt import BayesianOptimization
from typing import Iterable, Union
from tqdm import tqdm
from tqdm.contrib import itertools
from .experiment_instance import ExperimentInstance
import torch
from glob import glob
import json
import warnings
import math

warnings.filterwarnings("ignore", ".*does not have many workers.*")


torch.set_float32_matmul_precision('high')


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

        self.config_path = config_path
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
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


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
                bound_params = {}
                for key, value in general_params[general_key]['params'].items():
                    if type(value) == list:
                        if type(value[0]) == str or type(value[0]) == bool:
                            value = [0, len(value)-1]
                        bound_params[key] = value

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
                        if BuiltinClass==str or type(BuiltinClass) == bool:
                            params[general_key]['params'][key_to_update] = params[general_key]['params'][key_to_update][round(optimized_params[key_to_update])]
                        else:
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
        print(bounds)
        self.optimizer = BayesianOptimization(
            f=None,
            pbounds=bounds,
            verbose=2,
            random_state=1,
        )

        for _ in range(self.iterations):
            optimized_params = self.optimizer.suggest()      

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
    
    def remove_checkpoints(self, config_name):
        files = glob(f'checkpoints/{config_name}/*')
        for f in files:
            os.remove(f)

    def run(self) -> pd.DataFrame:
        """
        Run the experiment launcher.

        Returns:
            pd.DataFrame: Metrics DataFrame.
        """

        # Iterate over dataset, selection, and model configurations
        for dataset, selection, model in tqdm(
            itertools.product(self.data_configuration.keys(), 
                            self.selection_configuration.keys(), 
                            self.model_configuration.keys()), 
            desc="Processing Configurations"
        ):
            # Prepare parameters for dataset, selection, and model
            dataset_params = {"dataset": {'name': dataset, "params": self.data_configuration[dataset]}}
            selection_params = {"selection": {"name": selection, "params": self.selection_configuration[selection]}}
            model_params = {"model": {"name": model, "params": self.model_configuration[model]}}

            general_params = {**dataset_params, **selection_params, **model_params}
            self.seed()

            # Generate hyperparameter search space and iterate
            params_generator = tqdm(
                self.search_hyperparameters(general_params), 
                leave=False, 
                total=self.iterations, 
                desc=f"Dataset: {dataset}, Selection: {selection}, Model: {model}"
            )
            for params in params_generator:
                # Skip invalid combinations
                if (params['model']['params']['type'] == "sklearn" and params['selection']['name'] != 'NoSelection'):
                    continue

                # Construct experiment instance
                config_name = os.path.basename(self.config_path[:-1])
                experiment = ExperimentInstance(params, config_name)

                if self.is_performed(experiment, params):
                    continue

                # Update tqdm description to show JSON parameters prettily
                params_generator.set_description_str(f"Running with Params: {json.dumps(params, indent=1)}")

                # Ensure checkpoint directory exists
                if not os.path.exists(f'checkpoints/{config_name}'):
                    os.mkdir(f'checkpoints/{config_name}')

                self.remove_checkpoints(config_name)
                # Run the experiment and collect metrics
                metrics = experiment.run()

                # Register metrics with Bayesian optimizer if applicable
                if self.optimizer == 'bayesian':
                    self.optimizer.register(params=params, target=-metrics['root_mean_squared_error_valid'].mean())

                # Save metrics to the cumulative DataFrame and file
                self.metrics = pd.concat([self.metrics, metrics])
                self.metrics.to_csv(self.save_file, index=None)

                # Clean up checkpoints
                self.remove_checkpoints(config_name)

        return self.metrics