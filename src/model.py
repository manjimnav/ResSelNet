from tensorflow import keras
from tensorflow.keras import layers
from functools import partial
from .layer import TimeSelectionLayer, binary_sigmoid_unit
from sklearn.multioutput import MultiOutputRegressor
import numpy as np
from sklearn.base import BaseEstimator
import yaml
import importlib
import inspect
import os


def get_hyperparameters() -> tuple:
    """
    Get hyperparameters for the model.

    Returns:
        tuple: A tuple containing loss and metrics.
    """

    loss = keras.losses.MSE
    metrics = [keras.metrics.MSE, keras.metrics.MAE,
               keras.metrics.mean_absolute_percentage_error]

    return loss, metrics


def get_base_layer(layer_type: str) -> callable:
    """
    Get the base layer function based on the layer type.

    Args:
        layer_type (str): The type of layer ('dense', 'lstm', or 'cnn').

    Returns:
        callable: The base layer function.
    """
    if layer_type == 'dense':
        layer_base = layers.Dense
    elif layer_type == 'lstm':
        layer_base = layers.LSTM
    elif layer_type == 'cnn':
        layer_base = partial(layers.Conv1D, kernel_size=3)

    return layer_base

def head_layers(parameters: dict, n_features_out: int, name: str = '') -> list:
    """
    Create head layers based on the selection type in the parameters.

    Args:
        parameters (dict): The model parameters.
        n_features_out (int): Number of output features.
        name (str, optional): Name for the layers. Defaults to ''.

    Returns:
        list: List of head layers.
    """
    selection = parameters['selection']['name']
    
    head_layers = []
    if 'TimeSelectionLayer' in selection:
        regularization = parameters['selection']['params']['regularization']
        head_layers.append(TimeSelectionLayer(num_outputs=n_features_out,
                           regularization=regularization, name=f'{name}'))
    
    if parameters['model']['name'] == 'dense':
        head_layers.append(layers.Flatten())
    
    if len(head_layers)>0:
        return head_layers
    else:
        return None

def get_tf_model(parameters: dict, label_idxs: list, values_idxs: list) -> keras.Model:
    """
    Create a TensorFlow model based on the given parameters.

    Args:
        parameters (dict): The model parameters.
        label_idxs (list): List of label indices.
        values_idxs (list): List of value indices.

    Returns:
        keras.Model: The TensorFlow model.
    """
    model = parameters['model']['name']
    n_layers = parameters['model']['params']['layers']
    n_units = parameters['model']['params']['units']
    dropout = parameters['model']['params']['dropout']
    lr = parameters['model']['params']['lr']
    pred_len = parameters['dataset']['params']['pred_len']
    seq_len = parameters['dataset']['params']['seq_len']
    residual = parameters['selection'].get('params', dict()) or dict()
    residual = residual.get('residual', False)
    
    loss, metrics = get_hyperparameters()

    n_features_in = len(label_idxs) + len(values_idxs)
    n_features_out = len(label_idxs)
        
    layer_base = get_base_layer(model)
    
    inputs_raw = layers.Input(shape=(seq_len*n_features_in,), name='inputs')
    inputs = layers.Reshape((seq_len, n_features_in), name='inputs_reshaped')(inputs_raw)
    
    header = keras.Sequential(head_layers(parameters, n_features_out*pred_len, name=f'0'))
    
    x = inputs if header is None else header(inputs)
    
    for i in range(n_layers):

        if i > 0 and residual:
            header = keras.Sequential(head_layers(parameters, n_features_out*pred_len, name=f'{i+1}'))
            formatted_inputs = inputs if header is None else header(inputs)
        
            x = layers.Concatenate()([x, formatted_inputs])

        if model == 'lstm' and i < n_layers-1:
            kargs = {"return_sequences": True}
        else:
            kargs = {}

        x = layer_base(n_units, activation="relu" if model != 'lstm' else "tanh", name=f"layer{i}", **kargs)(x)
        x = layers.Dropout(dropout)(x)

    if residual:
        header = keras.Sequential(head_layers(parameters, n_features_out*pred_len, name=f'{n_layers+1 if n_layers>1 else n_layers}'))
        formatted_inputs = inputs if header is None else header(inputs)

        x = layers.Concatenate()([x, layers.Flatten()(formatted_inputs)])

    outputs = layers.Dense(n_features_out*pred_len, name="output")(x)
    model = keras.Model(inputs=inputs_raw, outputs=outputs, name="tsmodel")
        
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=loss,
        metrics=metrics
    )
    
    return model


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


def get_model(parameters: dict, label_idxs: list, values_idxs: list):
    """
    Create a model based on the given parameters.

    Args:
        parameters (dict): The model parameters.
        label_idxs (list): List of label indices.
        values_idxs (list): List of value indices.

    Returns:
        object: The model.
    """

    model_type = parameters['model']['params']['type']

    if model_type == 'tensorflow':
        model = get_tf_model(parameters, label_idxs, values_idxs)
    else:
        model = get_sk_model(parameters)

    return model


def get_selected_idxs(model: keras.Model, features: np.ndarray) -> set:
    """
    Get selected indices from the model's selection layers.

    Args:
        model (keras.Model): The TensorFlow model.
        features (np.ndarray): Input features.

    Returns:
        set: Set of selected indices.
    """
    
    selected_idxs = set()
    for layer in model.layers:
        if 'selection' in layer.name:
            mask = binary_sigmoid_unit(layer.get_mask()).numpy()
            selected_idxs = selected_idxs.union(np.arange(0, features.flatten().shape[0])[
                mask.flatten().astype(bool)].tolist())
        elif type(layer) == keras.Sequential:
            selected_idxs = selected_idxs.union(get_selected_idxs(layer, features))
    return selected_idxs
