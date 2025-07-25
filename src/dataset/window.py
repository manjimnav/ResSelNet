import tensorflow as tf
import numpy as np
from functools import partial

def collate_pair(x: tf.Tensor, pred_len: int, values_idxs: list, label_idxs: list) -> tuple:
    """
    Collate input data into pairs of selected inputs and corresponding outputs.

    Args:
        x (tf.Tensor): Input data.
        pred_len (int): Prediction length.
        values_idxs (list): Indices of value columns.
        label_idxs (list): Indices of label columns.
        selection_idxs (tf.Tensor, optional): Indices of selected features. Defaults to None.
        keep_dims (bool, optional): Whether to keep dimensions when selecting features. Defaults to False.

    Returns:
        tuple: Selected inputs and outputs as tensors.
    """
    seq_len = len(x)-pred_len
    inputs = x[:-pred_len]

    feat_size = len(label_idxs) + len(values_idxs)

    selected_inputs = tf.reshape(inputs, [seq_len*feat_size])
    reshaped_outputs = tf.reshape(x[-pred_len:], [pred_len, feat_size])

    outputs = tf.squeeze(tf.reshape(
        tf.gather(reshaped_outputs, [label_idxs], axis=1), [pred_len*len(label_idxs)]))
    return selected_inputs, outputs


def batch(seq_len: int, x: tf.Tensor) -> tf.Tensor:
    """
    Batch the input data with a specified sequence length.

    Args:
        seq_len (int): Sequence length.
        x (tf.Tensor): Input data.

    Returns:
        tf.Tensor: Batches of data.
    """
    return x.batch(seq_len)


def dataset_to_numpy_array(dataset):

    X_values = []
    y_values = []
    for elem in dataset.as_numpy_iterator():
        X_values.append(elem[0])
        y_values.append(elem[1])
    
    X_values = np.stack(X_values)
    y_values = np.stack(y_values)
    numpy_array = [X_values, y_values]

    return numpy_array


def divide_window(data_windowed, seq_len, pred_len, values_idxs, label_idxs, convert_to_numpy):

    batch_seq = partial(batch, seq_len+pred_len)

    data_windowed = data_windowed.flat_map(batch_seq).map(lambda x: collate_pair(x, pred_len, values_idxs, label_idxs))
        
    if convert_to_numpy:
        with tf.device('/cpu:0'):
            data_windowed = dataset_to_numpy_array(data_windowed)

    return data_windowed

def window_one(data_scaled, parameters, values_idxs: list, label_idxs: list, convert_to_numpy=True):

    seq_len = parameters['dataset']['params']['seq_len']
    pred_len = parameters['dataset']['params']['pred_len']
    shift = parameters['dataset']['params']['shift'] or seq_len
    
    if type(data_scaled)==tuple:
        data_windowed = tf.data.Dataset.from_tensor_slices(np.array([], dtype=np.float64)).window(size=1)
        groups = []
        for tsgroup, test_group in data_scaled:
            group_windowed = tf.data.Dataset.from_tensor_slices(test_group).window(seq_len+pred_len, shift=shift, drop_remainder=True)

            data_windowed = data_windowed.concatenate(group_windowed)

            groups.extend([tsgroup for _ in range(group_windowed.cardinality().numpy())])
        
        data_windowed = divide_window(data_windowed, seq_len, pred_len, values_idxs, label_idxs, convert_to_numpy)

        data_windowed = {"data": data_windowed, "groups": groups}
    else:
        data_windowed = tf.data.Dataset.from_tensor_slices(data_scaled)    
    
        data_windowed = data_windowed.window(seq_len+pred_len, shift=shift, drop_remainder=True)
        
        data_windowed = divide_window(data_windowed, seq_len, pred_len, values_idxs, label_idxs, convert_to_numpy)

        data_windowed = {"data": data_windowed}
    
    return data_windowed

def windowing(train_scaled: np.ndarray, valid_scaled: np.ndarray, test_scaled: np.ndarray, values_idxs: list, label_idxs: list, parameters: dict) -> tuple:
    """
    Prepare the data for windowing and batching.

    Args:
        train_scaled (np.ndarray): Scaled training dataset.
        valid_scaled (np.ndarray): Scaled validation dataset.
        test_scaled (np.ndarray): Scaled test dataset.
        values_idxs (list): Indices of value columns.
        label_idxs (list): Indices of label columns.
        selection_idxs (tf.Tensor): Indices of selected features.
        parameters (dict): Model parameters.

    Returns:
        tuple: Training, validation, and test datasets in the specified format.
    """
    model_type = parameters['model']['params']['type']
    batch_size = parameters['model']['params'].get('batch_size', 32)

    data_train = window_one(train_scaled, parameters, values_idxs, label_idxs, convert_to_numpy=False)
    data_valid = window_one(valid_scaled, parameters, values_idxs, label_idxs)
    data_test = window_one(test_scaled, parameters, values_idxs, label_idxs)


    if model_type == 'tensorflow':
        data_train["data"] = data_train["data"].batch(
            batch_size, drop_remainder=True).cache().prefetch(tf.data.AUTOTUNE)

    else:
        with tf.device('/cpu:0'):

            data_train["data"] = dataset_to_numpy_array(data_train["data"].shuffle(buffer_size=len(train_scaled), seed=123))

    
    return data_train, data_valid, data_test