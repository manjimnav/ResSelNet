from .scikit_factory import get_sk_model
from .tf_factory import get_tf_model
from .tf_models import get_selected_idxs
from typing import Tuple, Union, Optional
import tensorflow as tf
from sklearn.base import BaseEstimator

def get_model(parameters: dict, n_features_in, n_features_out):
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
        model = get_tf_model(parameters, n_features_in, n_features_out)
    else:
        model = get_sk_model(parameters)

    return model

__all__ = ["get_model", "get_sk_model", "get_tf_model", "get_selected_idxs"]
