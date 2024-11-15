
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Union, Type

class GroupByScaler(BaseEstimator, TransformerMixin):
    def __init__(self, BASE_SCALER: Type[TransformerMixin] =StandardScaler):
        self.scalers = dict()
        self.BASE_SCALER = BASE_SCALER
    
    def fit(self, X_groups):
        for group, data in X_groups:
            
            self.scalers[group] = self.BASE_SCALER().fit(data)

        return self
    
    def transform(self, X_groups):
        result = tuple()
        for group, data in X_groups:

            data = self.scalers[group].transform(data)

            result += ((group, data), )
            
        return result


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

def inverse_scale(y_vals, groups=None, scaler=None, label_idxs=None):

        iscaled_values = ()

        if groups is None:
            mean = scaler.mean_[label_idxs]
            std = scaler.scale_[label_idxs]

            if len(label_idxs)>1:
                mean = np.reshape(mean, (1, 1, len(mean)))
                std = np.reshape(std, (1, 1, len(std)))        

            for y in y_vals:

                if len(label_idxs)>1:
                    y = np.reshape(y, (y.shape[0], -1, len(label_idxs)))
                y_inverse = y*std + mean

                y_inverse = np.reshape(y_inverse, (y.shape[0], -1))
                iscaled_values = iscaled_values + (y_inverse,)
        else:
            for y_scaled in y_vals:
                y_inverse = []
                for group, y_val in zip(groups, y_scaled):
                    mean = scaler.scalers[group].mean_[label_idxs]
                    std = scaler.scalers[group].scale_[label_idxs]
                    if len(label_idxs)>1:
                        mean = np.reshape(mean, (mean.shape[0], -1, len(label_idxs)))
                        std = np.reshape(std, (std.shape[0], -1, len(label_idxs)))
                        y_val = np.reshape(y_val, (y_val.shape[0], -1, len(label_idxs)))

                    y_val_inverse = y_val*std + mean
                    y_val_inverse = np.reshape(y_val_inverse, (y.shape[0], -1))
                    y_inverse.append(y_val_inverse)

                iscaled_values = iscaled_values + (y_inverse,)
        
        return iscaled_values