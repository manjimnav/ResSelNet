from sklearn.multioutput import MultiOutputRegressor
import numpy as np
from sklearn.base import BaseEstimator
import yaml
import importlib
import inspect
import os

def get_sk_model(parameters: dict) -> BaseEstimator:

    directory = os.getcwd()
    available_models_config = yaml.safe_load(open(f'{directory}/src/skmodels.yaml', 'r')) 

    try:
        model_config = available_models_config[parameters['model']['name']]
        model_name, import_module, model_params = model_config['name'], model_config['module'], model_config.get('args', {})
        MODEL_CLASS = getattr(importlib.import_module(import_module), model_name)
    
    except Exception as e:
        print(e)
        raise NotImplementedError("Model not found or installed.")

    model_inspect = inspect.getfullargspec(MODEL_CLASS)
    if model_inspect.kwonlydefaults is not None:
        arguments = list(model_inspect.kwonlydefaults.keys())
        if 'random_state' in arguments:
            model_params.update({'random_state':123})
        if 'n_jobs' in arguments:
            model_params.update({'n_jobs':-1})

    
    model_params.update({k:v for k,v in parameters['model']['params'].items() if k != "type"})

    model = MODEL_CLASS(**model_params)
    
    if parameters['model']['name'] in ["svr", "gbr"]:
        model = MultiOutputRegressor(model, n_jobs=-1)
    
    return model