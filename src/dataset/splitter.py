import numpy as np


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