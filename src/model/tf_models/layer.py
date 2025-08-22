import tensorflow as tf
import math
from functools import partial
from typing import Type
import numpy as np
from tensorflow import keras
from tensorflow.keras import initializers, layers

from tensorflow import keras
from tensorflow.keras import layers

from .itransformer import (
    ChannelTransformerBlock,
    InstanceNormSeries,
    SeriesEmbedding,
    ChannelSinusoidalPE,
)

class TCNResidualBlock(layers.Layer):
    def __init__(self, filters, kernel_size=3, dilation_rate=1, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.conv = layers.Conv1D(
            filters, kernel_size,
            padding="causal",
            dilation_rate=dilation_rate,
            activation="relu",
            kernel_initializer=initializers.HeUniform(seed=123),
            bias_initializer=initializers.Zeros()
        )
        self.drop = layers.Dropout(dropout)
        self.proj = layers.Conv1D(
            filters, 1, padding="same",
            kernel_initializer=initializers.HeUniform(seed=123),
            bias_initializer=initializers.Zeros()
        )

    def call(self, x, training=False):
        y = self.conv(x)
        y = self.drop(y, training=training)
        if x.shape[-1] != y.shape[-1]:
            x = self.proj(x)
        return tf.nn.relu(x + y)

class TransformerEncoderBlock(layers.Layer):
    def __init__(self, d_model, num_heads=4, dff=2, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.d_model   = d_model
        # keep heads reasonable vs. d_model
        self.num_heads = max(1, min(num_heads, self.d_model))
        self.key_dim   = max(1, self.d_model // self.num_heads)
        self.dropout   = dropout

        # input projection if channels != d_model
        self.proj_in = layers.Dense(self.d_model)

        self.ln1 = layers.LayerNormalization(epsilon=1e-5)
        self.ln2 = layers.LayerNormalization(epsilon=1e-5)

        self.mha = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            output_shape=self.d_model,
            dropout=self.dropout,
        )
        self._needs_proj_out = False


        self.do1 = layers.Dropout(self.dropout)
        self.ffn = keras.Sequential([
            layers.Dense(int(dff), activation="relu"),
            layers.Dense(self.d_model),
        ])
        self.do2 = layers.Dropout(self.dropout)

    def _add_sinusoidal_pe(self, x):
        T = tf.shape(x)[1]
        d = self.d_model
        half = d // 2

        positions = tf.cast(tf.range(T)[:, None], tf.float32)
        div_term  = tf.pow(10000.0, -tf.range(half, dtype=tf.float32) / float(half))
        angles    = positions * div_term[None, :]

        pe_sin = tf.sin(angles)
        pe_cos = tf.cos(angles)
        pe     = tf.reshape(tf.stack([pe_sin, pe_cos], axis=-1), (T, -1))

        if d % 2 == 1:
            pe = tf.pad(pe, [[0, 0], [0, 1]])

        pe = pe[None, :, :]
        return x + pe

    def call(self, x, training=False):
        # ensure channel dim == d_model
        if x.shape[-1] != self.d_model:
            x = self.proj_in(x)
        
        x = self._add_sinusoidal_pe(x)

        qkv = self.ln1(x)
        y = self.mha(qkv, qkv, qkv, training=training)

        x = x + self.do1(y, training=training)

        y2 = self.ffn(self.ln2(x))
        x  = x + self.do2(y2, training=training)
        return x

def hard_sigmoid(x: tf.Tensor) -> tf.Tensor:
    """
    Compute the hard sigmoid activation function.

    Args:
        x (tf.Tensor): Input tensor.

    Returns:
        tf.Tensor: Output tensor with values in the range [0, 1].
    """
    return tf.keras.activations.hard_sigmoid(x)

def round_through(x: tf.Tensor) -> tf.Tensor:
    '''Element-wise rounding to the closest integer with full gradient propagation.
    A trick from [Sergey Ioffe](http://stackoverflow.com/a/36480182)
    a op that behave as f(x) in forward mode,
    but as g(x) in the backward mode.
    Args:
        x (tf.Tensor): Input tensor.

    Returns:
        tf.Tensor: Rounded tensor with gradient propagation.
    '''
    rounded = tf.round(x)
    return x + tf.stop_gradient(rounded-x)
  
def binary_sigmoid_unit(x: tf.Tensor) -> tf.Tensor:
    """
    Compute the binary sigmoid unit using rounding with gradient propagation.

    Args:
        x (tf.Tensor): Input tensor.

    Returns:
        tf.Tensor: Binary sigmoid unit output.
    """
    return round_through(hard_sigmoid(x))

class TimeSelectionLayer(tf.keras.layers.Layer):
    """
    Custom TensorFlow Keras layer for time selection.

    Args:
        num_outputs (int): Number of output units.
        regularization (float, optional): Regularization strength. Defaults to 0.001.
        **kwargs: Additional layer arguments.
    """
    def __init__(self, num_outputs: int, regularization: float = 0.001, flatten=False, **kwargs):
        super(TimeSelectionLayer, self).__init__( **kwargs)
        self.mask = None
        self.num_outputs = num_outputs
        self.regularization = regularization
        self.flatten = flatten
    
    def custom_regularizer(self, weights: tf.Tensor) -> tf.Tensor:
        """
        Custom regularization function for the layer.

        Args:
            weights (tf.Tensor): Layer weights.

        Returns:
            tf.Tensor: Regularization term.
        """
        weight = self.regularization/(10**math.log2(self.num_outputs))
        return tf.reduce_sum(weight * binary_sigmoid_unit(weights))

    def build(self, input_shape: tuple):
        if len(input_shape)>2:
            shape = [int(input_shape[-2]), int(input_shape[-1])]
        else:
            shape = [int(input_shape[-1])]

        self.mask = self.add_weight("kernel",
                                      shape=shape,
                                      initializer=tf.keras.initializers.Constant(value=0.01),
                                      regularizer=self.custom_regularizer)
        
    def get_mask(self):
        
        return binary_sigmoid_unit(tf.expand_dims(self.mask, 0))[0]
        
    def call(self, inputs: tf.Tensor) -> tf.Tensor:

        inputs_masked = tf.multiply(inputs, self.get_mask())

        if self.flatten:
            inputs_masked = tf.keras.layers.Flatten()(inputs_masked)
        
        return inputs_masked
    

def get_time_selection_layer(parameters: dict, n_features_out: int, flatten=False, name: str = '') -> list:
    """
    Instantiate the selection layer based on the selection type in the parameters.

    Args:
        parameters (dict): The model parameters.
        n_features_out (int): Number of output features.
        name (str, optional): Name for the layers. Defaults to ''.

    Returns:
        list: List of selection layer including the flatten if necessary.
    """
    
    regularization = parameters['selection']['params']['regularization']
    tsl = TimeSelectionLayer(num_outputs=n_features_out, regularization=regularization, flatten=flatten,  name=f'{name}')

    return tsl

def get_base_layer(layer_type: str) -> Type[tf.keras.layers.Layer]:
    """
    Get the base layer function based on the layer type.

    Args:
        layer_type (str): The type of layer ('dense', 'lstm', or 'cnn').

    Returns:
        callable: The base layer function.
    """
    if layer_type == 'dense':
        BASE_LAYER = partial(tf.keras.layers.Dense, activation='relu', kernel_initializer=initializers.GlorotUniform(seed=123), bias_initializer=initializers.Zeros())
    elif layer_type == 'lstm':
        BASE_LAYER = partial(tf.keras.layers.LSTM, activation='tanh', kernel_initializer=initializers.HeUniform(seed=123), bias_initializer=initializers.Zeros())
    elif layer_type == 'cnn':
        BASE_LAYER = partial(tf.keras.layers.Conv1D, padding='same', activation='relu', kernel_initializer=initializers.HeUniform(seed=123), bias_initializer=initializers.Zeros())

    elif layer_type == 'tcn':
        BASE_LAYER = TCNResidualBlock

    elif layer_type == 'transformer':
        BASE_LAYER = TransformerEncoderBlock

    elif layer_type == 'itransformer':
        BASE_LAYER = ChannelTransformerBlock

    return BASE_LAYER

def get_selected_idxs(model: tf.keras.Model, features: np.ndarray) -> set:
    """
    Get selected indices from the model's selection layers.

    Args:
        model (keras.Model): The TensorFlow model.
        features (np.ndarray): Input features.

    Returns:
        set: Set of selected indices.
    """
    
    selected_idxs = {}
    features_indexes = np.arange(0, features.flatten().shape[0])
    for layer in model.layers:
        if 'tsl' in layer.name:
            mask = binary_sigmoid_unit(layer.get_mask()).numpy()
            selected_idxs[layer.name] = features_indexes[mask.flatten().astype(bool)].tolist()
        elif type(layer) == tf.keras.Sequential:
            selected_idxs.update(get_selected_idxs(layer, features))
            
    return selected_idxs