
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import numpy as np
from functools import partial
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Union


class GroupByScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scalers = dict()
    
    def fit(self, X_groups):
        for group, data in X_groups:
            
            self.scalers[group] = StandardScaler().fit(data)
        return self
    
    def transform(self, X_groups):
        result = tuple()
        for group, data in X_groups:

            data = self.scalers[group].transform(data)

            result += ((group, data), )
            
        return result
    
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


def get_values_and_labels_index(data: np.ndarray) -> tuple:
    """
    Get the indices of value and label columns in the input data.

    Args:
        data (np.ndarray): Input data.

    Returns:
        tuple: Indices of label columns and value columns.
    """
    label_idxs = [idx for idx, col in enumerate(
        data.drop(['year', 'tsgroup'], axis=1, errors='ignore').columns) if 'target' in col]
    values_idxs = [idx for idx, col in enumerate(
        data.drop(['year', 'tsgroup'], axis=1, errors='ignore').columns) if 'target' not in col]

    return label_idxs, values_idxs


def scale(train_df: Union[np.ndarray, tuple], valid_df: Union[np.ndarray, tuple], test_df: Union[np.ndarray, tuple]) -> tuple:
    """
    Scale the input datasets using StandardScaler.

    Args:
        train_df (np.ndarray): Training dataset.
        valid_df (np.ndarray): Validation dataset.
        test_df (np.ndarray): Test dataset.

    Returns:
        tuple: Scaled training, validation, and test datasets, and the scaler object.
    """

    if type(train_df)==tuple:
        scaler = GroupByScaler()
    else:
        scaler = StandardScaler()

    train_scaled = scaler.fit_transform(train_df)
    valid_scaled = scaler.transform(valid_df)
    test_scaled = scaler.transform(test_df)

    return train_scaled, valid_scaled, test_scaled, scaler

def split_by_year(data, input_columns, test_year):
    val_year = test_year-1
    train_df = data.loc[~data.year.isin([test_year, val_year]), input_columns].values
    valid_df = data.loc[data.year == val_year, input_columns].values
    test_df = data.loc[data.year == test_year, input_columns].values

    return train_df, valid_df, test_df

def split_by_index(data, input_columns, train_val_index=(0.8, 0.1)):

    train_i, val_i = train_val_index

    data = data.loc[:, input_columns]

    if type(train_i)!=int:
        train_end_index = int(len(data)*train_i)
    else:
        train_end_index = train_i

    if type(val_i)!=int and type(train_i)!=int:
        val_end_indes = int(len(data)*(train_i+val_i))
    else:
        val_end_indes = val_i

    train_df = data.iloc[:train_end_index].values
    valid_df = data.iloc[train_end_index:val_end_indes].values
    test_df = data.iloc[val_end_indes:].values

    return train_df, valid_df, test_df
    

def split(data: np.ndarray, parameters: dict) -> tuple:
    """
    Split the data into training, validation, and test datasets based on the given parameters.

    Args:
        data (np.ndarray): Input data.
        parameters (dict): Model parameters.

    Returns:
        tuple: Training, validation, and test datasets.
    """

    input_columns = [col for col in data.columns.tolist() if col != 'year']

    test_year = parameters['dataset']['params'].get('test_year', None)
    seq_len = parameters['dataset']['params']['seq_len']
    pred_len = parameters['dataset']['params']['pred_len']

    if 'tsgroup' in data.columns.tolist():

        train_df, valid_df, test_df = tuple(), tuple(), tuple()

        for tsgroup, group_df in data.groupby("tsgroup"):
            
            group_df = group_df.drop("tsgroup", axis=1)

            if test_year!=None:
                train_df_group, valid_df_group, test_df_group = split_by_year(group_df, input_columns, test_year)
            else:
                train_end_index = min(int(len(group_df)*0.8), len(group_df)-seq_len*2)
                val_end_index = max(seq_len+pred_len, int(len(group_df)*0.9)-train_end_index)

                train_df_group, valid_df_group, test_df_group = split_by_index(group_df, input_columns, train_val_index=(train_end_index, val_end_index))

            train_df += ((tsgroup, train_df_group.values),)
            valid_df += ((tsgroup, valid_df_group.values),)
            test_df += ((tsgroup, test_df_group.values),)

    if test_year != None:

        train_df, valid_df, test_df = split_by_year(data, input_columns, test_year)

    else:
        train_df, valid_df, test_df = split_by_index(data, input_columns)


    return train_df, valid_df, test_df

def divide_window(data_windowed, seq_len, pred_len, values_idxs, label_idxs, convert_to_numpy):

    batch_seq = partial(batch, seq_len+pred_len)

    data_windowed = data_windowed.flat_map(batch_seq).map(lambda x: collate_pair(x, pred_len, values_idxs, label_idxs))
        
    if convert_to_numpy:
        data_windowed = list(map(lambda x: x.numpy(), next(data_windowed.batch(999999).__iter__())))
    
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
    batch_size = parameters['model']['params']['batch_size']

    data_train = window_one(train_scaled, parameters, values_idxs, label_idxs, convert_to_numpy=False)
    data_valid = window_one(valid_scaled, parameters, values_idxs, label_idxs)
    data_test = window_one(test_scaled, parameters, values_idxs, label_idxs)


    if model_type == 'tensorflow':
        data_train["data"] = data_train["data"].batch(
            batch_size, drop_remainder=True).cache().prefetch(tf.data.AUTOTUNE)

    else:
        data_train["data"] = list(map(lambda x: x.numpy(), next(
            data_train["data"].shuffle(buffer_size=len(train_scaled), seed=123).batch(999999).__iter__())))

    
    return data_train, data_valid, data_test


def get_feature_names(data, parameters):

    seq_len = parameters['dataset']['params']['seq_len']

    feature_names = np.array(
        [col for col in data.drop(['year', 'tsgroup'], axis=1, errors='ignore').columns])

    features = np.array([np.core.defchararray.add(
        feature_names, ' t-'+str(i)) for i in range(seq_len, 0, -1)]).flatten()

    return features
