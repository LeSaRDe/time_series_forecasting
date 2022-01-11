import json
import logging
import os
import sys
import time
from os import path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn


########################################################################################################################
#   HARD-CODED ZONE
#   TODO
#       Everything in this zone is merely for testing, and should be removed for official runs.
########################################################################################################################
g_task_name = 'weather'
g_work_folder = '/home/mf3jh/workspace/data/ml_test/'
g_raw_data_file_name = 'jena_climate_2009_2016.csv'
########################################################################################################################


########################################################################################################################
#   TASK SPECIFIC ZONE
#   TODO
#       Code in this zone is all task specific.
########################################################################################################################
def preprocess_weather_data(hourly=True):
    """
    Header:
    "Date Time","p (mbar)","T (degC)","Tpot (K)","Tdew (degC)","rh (%)","VPmax (mbar)","VPact (mbar)","VPdef (mbar)",
    "sh (g/kg)","H2OC (mmol/mol)","rho (g/m**3)","wv (m/s)","max. wv (m/s)","wd (deg)"
    Return:
        Pandas DataFrame
        Index:
            Time points
        Columns:
            Features
    """
    print('[preprocess_weather_data] Starts.')

    df_weather = pd.read_csv(g_data_settings['raw_data_file_name'])
    if hourly:
        df_weather = df_weather[5::6]

    wv = df_weather['wv (m/s)']
    wv[wv == -9999.0] = 0.0

    max_wv = df_weather['max. wv (m/s)']
    max_wv[max_wv == -9999.0] = 0.0

    wv = df_weather.pop('wv (m/s)')
    max_wv = df_weather.pop('max. wv (m/s)')

    # Convert to radians.
    wd_rad = df_weather.pop('wd (deg)') * np.pi / 180

    # Calculate the wind x and y components.
    df_weather['Wx'] = wv * np.cos(wd_rad)
    df_weather['Wy'] = wv * np.sin(wd_rad)

    # Calculate the max wind x and y components.
    df_weather['max Wx'] = max_wv * np.cos(wd_rad)
    df_weather['max Wy'] = max_wv * np.sin(wd_rad)

    date_time = pd.to_datetime(df_weather.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')
    timestamp_s = date_time.map(pd.Timestamp.timestamp)
    day = 24 * 60 * 60
    year = (365.2425) * day

    df_weather['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    df_weather['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
    df_weather['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    df_weather['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

    plot_cols = ['T (degC)', 'p (mbar)', 'rho (g/m**3)']
    plot_features = df_weather[plot_cols]
    plot_features.index = date_time
    _ = plot_features.plot(subplots=True)
    plt.show()
    plt.clf()
    plt.close()

    plot_features = df_weather[plot_cols][:480]
    plot_features.index = date_time[:480]
    _ = plot_features.plot(subplots=True)

    plt.show()
    plt.clf()
    plt.close()

    df_weather = df_weather.rename(columns={"Date Time": 'time',
                                            "p (mbar)": 'p',
                                            "T (degC)": 'T',
                                            "Tpot (K)": 'Tpot',
                                            "Tdew (degC)": 'Tdew',
                                            "rh (%)": 'rh',
                                            "VPmax (mbar)": 'VPmax',
                                            "VPact (mbar)": 'VPact',
                                            "VPdef (mbar)": 'VPdef',
                                            "sh (g/kg)": 'sh',
                                            "H2OC (mmol/mol)": 'H2OC',
                                            "rho (g/m**3)": 'rho',
                                            "max Wx": 'Wxmax',
                                            "max Wy": 'Wymax',
                                            "Day sin": 'day_sin',
                                            "Day cos": 'day_cos',
                                            "Year sin": 'year_sin',
                                            "Year cos": 'year_cos'})

    pd.to_pickle(df_weather, g_preprocessed_data_path)
    print('[preprocess_weather_data] All done.')
    return df_weather
########################################################################################################################


########################################################################################################################
#   GLOBAL PATH SETTINGS
########################################################################################################################
"""
The folder structure is as follows:
[g_work_folder]
    |--global_settings.json
    |--[g_task_name]
            |--conf
            |--raw_data
            |--preprocessed_data
            |--train_val_test
            |--models

'conf': All settings in JSON files.
'raw_data': All raw data, no matter of whatever format. 
'preprocessed_data': Pandas DataFrame pickle files. Preprocessed. Indexed over time (not necessarily consecutive). 
                     Columns are features. 
'train_val_test': NumPy npz files. Directly used for training, validation and testing. 
'models': Saved models. 
"""
if 'g_work_folder' not in globals():
    g_work_folder = os.getenv('ML_WORK_DIR')
    if g_work_folder is None:
        logging.error('[ENV] Please set up the environment variable "ML_WORK_DIR". '
                      'Otherwise, the current folder is used as the work folder.')
        g_work_folder = './'

# if 'g_task_name' not in globals():
#     g_task_name = os.getenv('ML_TASK_NAME')
#     if g_task_name is None:
#         logging.error('[ENV] Please set up the environment variable "ML_TASK_NAME".')
#         sys.exit(-1)

# if 'g_raw_data_file_name' not in globals():
#     g_raw_data_file_name = os.getenv('ML_RAW_DATA_FILE_NAME')
#     if g_raw_data_file_name is None:
#         logging.error('[ENV] Please set up the environment variable "ML_RAW_DATA_FILE_NAME".')
#         sys.exit(-1)

if not path.exists(path.join(g_work_folder, g_task_name)):
    logging.error('[ENV] %s does not exist.' % path.join(g_work_folder, g_task_name))
    sys.exit(-1)

# GLOBAL SETTINGS
g_global_settings_path = path.join(g_work_folder, 'global_settings.json')
if not path.exists(g_global_settings_path):
    logging.error('[ENV] %s does not exist.' % g_global_settings_path)
    sys.exit(-1)

# TASK SETTINGS
g_config_folder = path.join(g_work_folder, g_task_name, 'conf')
if not path.exists(g_config_folder):
    logging.error('[ENV] %s does not exist.' % g_config_folder)
    sys.exit(-1)
g_data_settings_file_path = path.join(g_config_folder, 'data_settings.json')
g_model_hyperparams_file_path = path.join(g_config_folder, 'model_hyperparams.json')
g_loss_func_hyperparams_file_path = path.join(g_config_folder, 'loss_func_hyperparams.json')
g_optimizer_hyperparams_file_path = path.join(g_config_folder, 'optimizer_hyperparams.json')
g_train_hyperparams_file_path = path.join(g_config_folder, 'train_hyperparams.json')
g_eval_hyperparams_file_path = path.join(g_config_folder, 'eval_hyperparams.json')
g_eval_func_hyperparams_file_path = path.join(g_config_folder, 'eval_func_hyperparams.json')

g_setting_name_to_file_path = \
    {
        'g_data_settings': g_data_settings_file_path,
        'g_model_hyperparams': g_model_hyperparams_file_path,
        'g_loss_func_hyperparams': g_loss_func_hyperparams_file_path,
        'g_optimizer_hyperparams': g_optimizer_hyperparams_file_path,
        'g_train_hyperparams': g_train_hyperparams_file_path,
        'g_eval_hyperparams': g_eval_hyperparams_file_path,
        'g_eval_func_hyperparams': g_eval_func_hyperparams_file_path
    }

# RAW DATA
g_raw_data_folder = path.join(g_work_folder, g_task_name, 'raw_data')
if not path.exists(g_raw_data_folder):
    logging.error('[ENV] %s does not exist.' % g_raw_data_folder)
    sys.exit(-1)

# g_raw_data_path = path.join(g_raw_data_folder, g_raw_data_file_name)
# if not path.exists(g_raw_data_path):
#     logging.error('[ENV] g_raw_data_path: %s does not exist.' % g_raw_data_path)
#     sys.exit(-1)

# PREPROCESSED DATA
g_preprocessed_data_folder = path.join(g_work_folder, g_task_name, 'preprocessed_data')
if not path.exists(g_preprocessed_data_folder):
    os.mkdir(g_preprocessed_data_folder)
g_preprocessed_data_path = path.join(g_preprocessed_data_folder, 'preprocessed_%s.pickle' % g_task_name)

# TRAIN, VALIDATION, TEST DATA
g_train_val_test_folder = path.join(g_work_folder, g_task_name, 'train_val_test')
if not path.exists(g_train_val_test_folder):
    os.mkdir(g_train_val_test_folder)

g_train_data_path = path.join(g_train_val_test_folder, 'train_%s.pickle' % g_task_name)
g_val_data_path = path.join(g_train_val_test_folder, 'val_%s.pickle' % g_task_name)
g_test_data_path = path.join(g_train_val_test_folder, 'test_%s.pickle' % g_task_name)

# The "input_label" data is more friendly for training. They are compressed npz files. Each contains two NumPy ndarrays.
# One is labeled as "np_input", and the other is labeled as "np_label". Use the function "load_input_label_data" to
# load in an "input_label" data.
g_train_input_label_path = path.join(g_train_val_test_folder, 'train_input_label_%s.npz' % g_task_name)
g_val_input_label_path = path.join(g_train_val_test_folder, 'val_input_label_%s.npz' % g_task_name)
g_test_input_label_path = path.join(g_train_val_test_folder, 'test_input_label_%s.npz' % g_task_name)

# MODELS
g_model_folder = path.join(g_work_folder, g_task_name, 'models')
if not path.exists(g_model_folder):
    os.mkdir(g_model_folder)

# RESULTS
g_results_folder = path.join(g_work_folder, g_task_name, 'results')
if not path.exists(g_results_folder):
    os.mkdir(g_results_folder)
########################################################################################################################


########################################################################################################################
#   UTILITY
########################################################################################################################
def purge_task(l_folders=['preprocessed', 'train_val_test', 'model', 'results']):
    """
    Remove everything except raw data and settings.
    """
    print('[purge_task] Starts.')
    if 'preprocessed' in l_folders:
        for file in os.listdir(g_preprocessed_data_folder):
            os.remove(path.join(g_preprocessed_data_folder, file))
            print('[purge_task] Purged g_preprocessed_data_folder.')

    if 'train_val_test' in l_folders:
        for file in os.listdir(g_train_val_test_folder):
            os.remove(path.join(g_train_val_test_folder, file))
            print('[purge_task] Purged g_train_val_test_folder.')

    if 'model' in l_folders:
        for file in os.listdir(g_model_folder):
            os.remove(path.join(g_model_folder, file))
            print('[purge_task] Purged g_model_folder.')

    if 'results' in l_folders:
        for file in os.listdir(g_results_folder):
            os.remove(path.join(g_results_folder, file))
            print('[purge_task] Purged g_results_folder.')

    print('[purge_task] All done.')
########################################################################################################################


########################################################################################################################
#   LOAD SETTINGS
########################################################################################################################
"""
In 'g_data_settings', each data set should have its own settings, and this set of settings is labeled by 
its 'g_task_name'. 
g_data_settings parameters:
:param
    input_seq_size: int (> 0)
        The length of each input time series fragment.
:param
    input_seq_stride: int (> 0)
        The number of skipped time points between two consecutive input sequences within a batch.
:param
    label_size: int (> 0)
        The length of time series to be predicted.
:param
    batch_size: int (> 0)
        The number of input sequences for each batch.
:param
    batch_stride: int (> 0)
        The number of skipped time points between two consecutive batches.
:param
    l_feature_cols: list of str or None
        If the value is None, use all columns as features.
        The list of column names for input features.
:param
    l_label_cols: list of str
        The list of column names for label features.

For example:
g_data_settings = \
    {
        'weather':
            {
                'input_seq_size': 14,
                'input_seq_stride': 3,
                'label_size': 1,
                'batch_size': 5,
                'batch_stride': 2,
                # 'l_feature_cols': ['p', 'T', 'Tpot', 'Tdew', 'rh', 'VPmax', 'VPact', 'VPdef', 'sh', 'H2OC', 'rho',
                #                    'Wx', 'Wy', 'Wxmax', 'Wymax', 'day_sin', 'day_cos', 'year_sin', 'year_cos'],
                'l_feature_cols': None,
                'l_label_cols': ['T']
            }
    }
"""

"""
Hyperparameters for models. Specified in 'g_model_hyperparams'. 

Global Hyperparameters:
:param
    num_in_feature: int (> 1)
        The number of features of each input time point. 
:param
    num_out_feature: int (> 1)
        The number of features of each predicted time point.
:param
    out_seq_size: int (> 1)
        The length of predicted time sequence.

Model Specific Hyperparameters:
@custom_lstm
:param
    lstm_hidden_size: int (> 1)
        The dimension of hidden states and cell states in LSTM. 
:param
    lstm_num_layers: int (> 1)
        The number of stacked LSTM layers. 
:param
    lstm_dropout: float (>= 0.0)
        The probability of dropout.

For example:
g_model_hyperparams = \
    {
        'global':
            {
                'num_in_feature': 19,
                'num_out_feature': 1,
                'out_seq_size': 1
            },
        'custom_lstm':
            {
                'lstm_hidden_size': 2,
                'lstm_num_layers': 1,
                'lstm_dropout': 0
            }
    }
"""

"""
Hyperparameters for loss functions. Specified in 'g_loss_func_hyperparams'.

For example:
g_loss_func_hyperparams = \
    {
        'mse':
            {
                'reduction': 'mean'
            }
    }
"""

"""
Hyperparameters for optimizers. Specified in 'g_optimizer_hyperparams'.

For example:
g_optimizer_hyperparams = \
    {
        'adagrad': {}
    }
"""

"""
Hyperparameters for training. Specified in 'g_train_hyperparams'.

:param
    'epoch': int (> 0)
        The number of iterations for training.
:param
    'train_device': str
        - 'cpu': Use CPU to train model
        - 'cuda': Use GPU to train model
:param
    model_name: str
        The model definition corresponding to 'model_name' is specified in 'build_model()'.
:param
    loss_func_name: str
        The loss function corresponding to 'loss_func_name' is specified in 'build_loss_func()'.
:param
    optimizer_name: str
        The optimizer corresponding to 'optimizer_name' is specified in 'build_optimizer()'.
:param
    train_set_name: str
        - 'train': Use the training set to train the model.
        - 'val': Use the validation set to train the model.
        - 'test': Use the test set to train the model.
:param
    num_top_checkpoint: int (> 0, default=10)
        The max number of top performance checkpoints to be saved.
:param
    epoch_ratio_for_save: float (>=0, default=0.5)
        The ratio of epochs to be skipped before starting to save checkpoints.

For example:
g_train_hyperparams = \
    {
        'epoch': 100,
        'train_device': 'cuda',
        'model_name': 'custom_lstm',
        'loss_func_name': 'mse',
        'optimizer_name': 'adagrad',
        'train_set_name': 'train',
        'num_top_checkpoint': 10,
        'epoch_ratio_for_save': 0.5
    }
"""

"""
Hyperparameters for evaluation. Specified in 'g_eval_hyperparams.'

:param
    'eval_func': str
        The name of evaluation metric function.
:param
    'eval_set_name': str
        The 

For example:
g_eval_hyperparams = \
    {
        'eval_func': 'nnse'
        'eval_set_name': 'test'
    }
"""

"""
Hyperparameters for evaluation functions. Specified in 'g_eval_func_hyperparams'.

For example:
g_eval_func_hyperparams = \
    {
        "nse":
        {
            "reduction": "mean",
            "time_point_metric": "l2_sqr",
            "normalized": false
        },
    
        "nnse":
        {
            "reduction": "mean",
            "time_point_metric": "l2_sqr",
            "normalized": true
        }
    }
"""


def load_settings(l_setting_names):
    """
    Load settings of names specified by 'l_setting_names'.
    :param
        l_setting_names: list of str
            Each setting name should a key of 'g_setting_name_to_file_path'.
    """
    print('[load_settings] Starts.')

    for setting_name in l_setting_names:
        if setting_name not in g_setting_name_to_file_path \
                or not path.exists(g_setting_name_to_file_path[setting_name]):
            raise Exception('[load_data_settings] %s cannot be loaded.' % setting_name)

        with open(g_setting_name_to_file_path[setting_name], 'r') as in_fd:
            setting_val = json.load(in_fd)
            globals()[setting_name] = setting_val
    print('[load_settings] All done.')
########################################################################################################################


########################################################################################################################
#   DATA PREPARATION
########################################################################################################################
"""
The "input_label" data should be a 4-dimensional array in the format:
('batch', 'sequence', 'time', 'feature')
In our definitions, the data record at each 'time' point is a real-valued vector, and each vector element corresponds to 
a 'feature' describing the data. A 'sequence' consists of a series of time points with data. And a 'batch' is a group of
sequences. 'batch' dim indicates how many batches are generated, 'sequence' dim indicates how many sequences contained
in a batch, 'time' dim indicates how many time points contained in each sequence, and 'feature' dim indicates how
many features are used to describe the data at each time point.

For example:
Given a time series: [[f1a, f1b, f1c], [f2a, f2b, f2c], ..., [f20a, f20b, f20c]], where
fnx represents the value of the xth feature at the time point n. In this example, there are n=20 time points, and
each time point is associated with 3 features named a, b and c. In a pandas DataFrame, this time series would look
like this:
    a    b    c
1  f1a  f1b  f1c
2  f2a  f2b  f2c
...
10 f10a f10b f10c
Suppose we set:
    input_seq_size = 3
    input_seq_stride = 2
    label_size = 2
    batch_size = 5
    batch_stride = 3
    l_feature_cols = ['a', 'b']
    l_label_cols = ['b']
what we desire looks like:
np_input:
    [
        # Batch 0
        [
            # Sequence 0-0
            [
                # Time 0-0-0
                [f1a, f1b],
                # Time 0-0-1
                [f2a, f2b],
                # Time 0-0-2
                [f3a, f3b]
            ],
            # Sequence 0-1
            # The 2nd seq starts with t=3 because of input_seq_stride = 2
            # i.e. skipping 2 time points from the starting time point of the 1st sequence.
            [[f3a, f3b], [f4a, f4b], [f5a, f5b]],
            # Sequence 0-2
            [[f5a, f5b], [f6a, f6b], [f7a, f7b]],
            # Sequence 0-3
            [[f7a, f7b], [f8a, f8b], [f9a, f9b]],
            # Sequence 0-4
            [[f9a, f9b], [f10a, f10b], [f11a, f11b]]
        ],
        # Batch 1
        [
            # Sequence 1-0
            # The 1st seq in this batch starts with time points [7, 8, 9] because of batch_stride = 3
            # i.e. skipping 3 sequences from the fist seq in Batch 0.
            [[f7a, f7b], [f8a, f8b], [f9a, f9b]],
            # Sequence 1-1
            [[f9a, f9b], [f10a, f10b], [f11a, f11b]],
            # Sequence 1-2
            [[f11a, f11b], [f12a, f12b], [f13a, f13b]],
            # Sequence 1-3
            [[f13a, f13b], [f14a, f14b], [f15a, f15b]],
            # Sequence 1-4
            [[f15a, f15b], [f16a, f16b], [f17a, f17b]]
        ]
    ]
np_label:
    [
        # Batch 0
        [
            [f4b, f5b],
            [f6b, f7b],
            [f8b, f9b],
            [f10b, f11b],
            [f12b, f13b]
        ],
        # Batch 1
        [
            [f10b, f11b],
            [f12b, f13b],
            [f14b, f15b],
            [f16b, f17b],
            [f18b, f19b]
        ]
    ]
    
How the data should be organized is specified in 'g_data_settings'. 
"""


def preprocess_data(data_name):
    print('[preprocess_data] Starts.')
    if data_name == 'weather':
        preprocess_weather_data()
    print('[preprocess_data] All done.')


def split_data(save_ret=True):
    """
    Split the preprocessed data specified by 'g_preprocessed_data_path' into 'train', 'val' and 'test' sets.
    Note that 'train_ratio' + 'val_ratio' must be less or equal to 1, and 'test_ratio' (not a parameter of this
    function) is induced by '1.0 - train_ratio - val_ratio'.
    """
    print('[split_data] Starts.')
    global g_data_settings
    if g_data_settings is None:
        raise Exception('[split_data] g_data_settings is None.')

    train_ratio = g_data_settings['train_ratio']
    val_ratio = g_data_settings['val_ratio']

    if train_ratio + val_ratio > 1.0:
        raise Exception('[split_data] train_ratio + val_ratio is greater than 1.0! train_ratio=%s, val_ratio=%s.'
                        % (train_ratio, val_ratio))

    if train_ratio + val_ratio == 1.0:
        logging.error('[split_data] train_ratio + val_ratio is equal 1.0! This setting will lead to an empty test set.')

    df_preprocessed = pd.read_pickle(g_preprocessed_data_path)

    num_rec = len(df_preprocessed)
    df_train = df_preprocessed[0:int(num_rec * train_ratio)]
    df_val = df_preprocessed[int(num_rec * train_ratio):int(num_rec * (train_ratio + val_ratio))]
    df_test = df_preprocessed[int(num_rec * (1 - train_ratio - val_ratio)):]

    train_mean = df_train.mean()
    train_std = df_train.std()

    df_train = (df_train - train_mean) / train_std
    df_val = (df_val - train_mean) / train_std
    df_test = (df_test - train_mean) / train_std

    if save_ret:
        pd.to_pickle(df_train, g_train_data_path)
        pd.to_pickle(df_val, g_val_data_path)
        pd.to_pickle(df_test, g_test_data_path)
        print('[build_inputs_and_labels] Results are saved.')

    print('[split_data] All done.')
    return df_train, df_val, df_test


def build_input_and_label_data(df_data, save_ret=False, out_path=None):
    """
    Construct the input-label data for a given DataFrame.
    :param
        df_data: pandas DataFrame
            Data source. Indexed by time points. Columns are features.
            NOTE: data preprocessing on df_data should have been done before it is sent here.
    :return
        np_input: NumPy 4-dim ndarray
            The input data sent to models.
    :return
        np_label: NumPy 4-dim ndarray
            The corresponding labels.
    """
    print('[build_input_and_label_data] Starts.')
    global g_data_settings
    if g_data_settings is None:
        raise Exception('[build_input_and_label_data] g_data_settings is None.')
    timer_start = time.time()

    input_seq_size = g_data_settings['input_seq_size']
    input_seq_stride = g_data_settings['input_seq_stride']
    label_size = g_data_settings['label_size']
    batch_size = g_data_settings['batch_size']
    batch_stride = g_data_settings['batch_stride']
    if g_data_settings['l_feature_cols'] is None:
        l_feature_cols = df_data.columns.to_list()
    else:
        l_feature_cols = g_data_settings['l_feature_cols']
    l_label_cols = g_data_settings['l_label_cols']

    l_full_seq = [df_data[i : i + input_seq_size + label_size] for i in
                  range(0, len(df_data) - (input_seq_size + label_size) + 1, input_seq_stride)]

    l_input_seq = []
    l_label_seq = []
    for i in range(0, len(l_full_seq) - batch_size + 1, batch_stride):
        input_batch = []
        label_batch = []
        for df_each in l_full_seq[i: i + batch_size]:
            input_batch.append(np.asarray(df_each[:input_seq_size][l_feature_cols]))
            label_batch.append(np.asarray(df_each[input_seq_size:][l_label_cols]))
        l_input_seq.append(np.stack(input_batch))
        l_label_seq.append(np.stack(label_batch))
    np_input = np.stack(l_input_seq)
    np_label = np.stack(l_label_seq)

    if save_ret:
        np.savez_compressed(out_path, np_input=np_input, np_label=np_label)
        print('[build_input_and_label_data] Results are saved.')

    print('[build_input_and_label_data] All done in %s secs.' % str(time.time() - timer_start))
    return np_input, np_label


def build_input_and_label_data_for_train_val_test():
    print('[build_input_and_label_data_for_train_val_test] Starts.')
    l_input_label_data_path = [g_train_input_label_path, g_val_input_label_path, g_test_input_label_path]
    for data_idx, data_path in enumerate([g_train_data_path, g_val_data_path, g_test_data_path]):
        df_data = pd.read_pickle(data_path)
        build_input_and_label_data(df_data, save_ret=True, out_path=l_input_label_data_path[data_idx])
    print('[build_input_and_label_data_for_train_val_test] All done.')


def load_input_label_data(ds_name):
    """
    Load in an "input_label" data, and return the input and label ndarrays.
    :param
        ds_name: str
            'train': Load from g_train_input_label_path
            'val': Load from g_val_input_label_path
            'test': Load from g_test_input_label_path
    """
    if ds_name == 'train':
        input_label_data_path = g_train_input_label_path
    elif ds_name == 'val':
        input_label_data_path = g_val_input_label_path
    elif ds_name == 'test':
        input_label_data_path = g_test_input_label_path
    else:
        raise Exception('[load_input_label_data] Invalid ds_name %s.' % ds_name)

    if not path.exists(input_label_data_path):
        raise Exception('[load_input_label_data] %s does not exist.' % input_label_data_path)

    input_label_data = np.load(input_label_data_path)
    np_input = input_label_data['np_input']
    np_label = input_label_data['np_label']
    return np_input, np_label
########################################################################################################################


########################################################################################################################
#   Train, Validation & Test
########################################################################################################################
def build_model(model_name):
    """
    Create a model instance. The hyperparameters of the model is specified in 'g_model_hyperparams'.
    :param
        model_name: str
            The name of the model.
    :return: torch.nn.Module
        The instance of the model.
    """
    global g_model_hyperparams
    if g_model_hyperparams is None:
        raise Exception('[build_model] g_model_hyperparams is None')
    num_in_feature = g_model_hyperparams['global']['num_in_feature']
    num_out_feature = g_model_hyperparams['global']['num_out_feature']
    out_seq_size = g_model_hyperparams['global']['out_seq_size']

    if model_name == 'custom_lstm':
        lstm_hidden_size = g_model_hyperparams['custom_lstm']['lstm_hidden_size']
        lstm_num_layers = g_model_hyperparams['custom_lstm']['lstm_num_layers']
        lstm_dropout = g_model_hyperparams['custom_lstm']['lstm_dropout']
        model = CustomLSTM(num_in_feature, num_out_feature, out_seq_size, lstm_hidden_size, lstm_num_layers, lstm_dropout)
    else:
        model = None

    return model


def build_loss_func(loss_func_name):
    """
    Create a loss function instance. The hyperparameters of the loss function is specified in 'g_loss_func_hyperparams'.
    :param
        loss_func_name: str
            The name of the loss function.
    :return: torch.nn.modules.loss._Loss
        The instance of the loss function.
    """
    global g_loss_func_hyperparams
    if loss_func_name == 'mse':
        if g_loss_func_hyperparams is None:
            raise Exception('[build_loss_func] g_loss_func_hyperparams is None.')
        reduction = g_loss_func_hyperparams['mse']['reduction']
        loss_func = nn.MSELoss(reduction=reduction)
    else:
        loss_func = None

    return loss_func


def build_optimizer(optimizer_name, params):
    """
    Create an optimizer instance. The hyperparameters of the optimizer is specified in 'g_optimizer_hyperparams'.
    :param
        optimizer_name: str
            The name of the optimizer.
    :param
        params: generator/list of torch.nn.parameter.Parameter
            The parameters to be optimized.
    :return: torch.optim.Optimizer
        The instance of the optimizer.
    """
    global g_optimizer_hyperparams
    if optimizer_name == 'adagrad':
        optimizer = torch.optim.Adagrad(params=params)
    else:
        optimizer = None

    return optimizer


def build_eval_func(eval_func_name):
    """
    Create an evaluation function instance.
    :param
        eval_func_name: str
            The name of the evaluation function.
    :return: torch.nn.modules.loss._Loss
        The instance of the evaluation function.
    """
    global g_eval_func_hyperparams
    if eval_func_name == 'nse':
        if g_eval_func_hyperparams is None:
            raise Exception('[build_eval_func] g_eval_func_hyperparams is None.')
        reduction = g_eval_func_hyperparams[eval_func_name]['reduction']
        time_point_metric = g_eval_func_hyperparams[eval_func_name]['time_point_metric']
        normalized = g_eval_func_hyperparams[eval_func_name]['normalized']
        eval_func = NSEloss(reduction=reduction, time_point_metric=time_point_metric, normalized=normalized)
    elif eval_func_name == 'nnse':
        if g_eval_func_hyperparams is None:
            raise Exception('[build_eval_func] g_eval_func_hyperparams is None.')
        reduction = g_eval_func_hyperparams[eval_func_name]['reduction']
        time_point_metric = g_eval_func_hyperparams[eval_func_name]['time_point_metric']
        normalized = g_eval_func_hyperparams[eval_func_name]['normalized']
        eval_func = NSEloss(reduction=reduction, time_point_metric=time_point_metric, normalized=normalized)
    else:
        eval_func = None
    return eval_func


def train_model(epoch_log_ratio=0.1):
    """
    Train the model specified by 'model_name', and save the trained models of top performance.
    :param
        epoch_log_ratio: float ([0, 1], default=0.1)
        Specifies the frequency of logging.
    :return: None
        Checkpoints will be saved. A checkpoint is a dict:
        {
            'loss': [the loss of training of the saved model]
            'model_state_dict': [model state_dict]
            'optimizer_state_dict': [optimizer state_dict]
            'train_hyperparams': [training hyperparameters]
            'model_global_hyperparams': [model global hyperparameters]
            'model_specific_hyperparams': [model specific hyperparameters]
            'loss_func_hyperparams': [loss function hyperparameters]
            'optimizer_hyperparams': [optimizer hyperparameters]
            'batch_loss_log': [loss values over all epochs of all batches]
        }
        Each checkpoint is named as follows:
        'model#[model name]#[optimizer name]#[loss function name]#[loss 2-digit precision]#[epoch]#[training device]#[time stamp]#[model index].pickle'
        For example:
        'model#custom_lstm#adagrad#mse#1.04#100#cuda#211208230608#0.pickle'
    """
    timer_init = time.time()
    print('[train_model] Starts.')
    global g_train_hyperparams
    if g_train_hyperparams is None:
        raise Exception('[train_model] g_train_hyperparams is None.')

    # RETRIEVE TRAINING HYPERPARAMETERS
    model_name = g_train_hyperparams['model_name']
    loss_func_name = g_train_hyperparams['loss_func_name']
    optimizer_name = g_train_hyperparams['optimizer_name']
    epoch = g_train_hyperparams['epoch']
    train_set_name = g_train_hyperparams['train_set_name']
    num_top_checkpoint = g_train_hyperparams['num_top_checkpoint']
    epoch_ratio_for_save = g_train_hyperparams['epoch_ratio_for_save']
    train_device = g_train_hyperparams['train_device']

    epoch_log_batch = int(epoch_log_ratio * epoch)

    # LOAD TRAINING DATA
    np_input, np_label = load_input_label_data(train_set_name)

    # l_top_models format:
    #   [(loss, model_state_dict), ...]
    l_top_models = []
    min_model_idx = 0

    # CREATE MODEL, LOSS FUNCTION & OPTIMIZER
    model = build_model(model_name)
    loss_func = build_loss_func(loss_func_name)
    optimizer = build_optimizer(optimizer_name, model.parameters())

    # DETERMINE DEVICE
    if train_device == 'cuda' and not torch.cuda.is_available():
        logging.error('[train_model] GPU is not available. Use CPU instead.')
        train_device = 'cpu'

    # TRAINING
    l_batch_loss_rec = []
    timer_start = time.time()
    with torch.cuda.device(train_device):
        # for batch_i in range(np_input.shape[0]):
        for batch_i in range(2):
            l_loss_rec = []
            th_batch_input = torch.from_numpy(np_input[batch_i]).type(torch.float32)
            th_batch_label = torch.from_numpy(np_label[batch_i]).type(torch.float32)

            for epoch_i in range(epoch):
                th_output = model(th_batch_input)
                optimizer.zero_grad()
                loss = loss_func(th_output, th_batch_label)
                loss.backward()
                optimizer.step()
                l_loss_rec.append((epoch_i, loss.item()))
                if epoch_i % epoch_log_batch == 0:
                    print('[train_model] epoch %s: loss=%1.5f, elapse=%s'
                          % (epoch_i, loss.item(), time.time() - timer_start))
                if (epoch_i + 1) / epoch >= epoch_ratio_for_save:
                    if len(l_top_models) < num_top_checkpoint:
                        model_state_dict = model.state_dict()
                        optimizer_state_dict = optimizer.state_dict()
                        l_top_models.append((loss.item(), model_state_dict, optimizer_state_dict, epoch_i + 1))
                    else:
                        if loss.item() < l_top_models[min_model_idx][0]:
                            model_state_dict = model.state_dict()
                            optimizer_state_dict = optimizer.state_dict()
                            l_top_models[min_model_idx] = (loss.item(), model_state_dict, optimizer_state_dict,
                                                           epoch_i + 1)
                    min_model_idx = np.argmax([item[0] for item in l_top_models])
            df_loss_log = pd.DataFrame(l_loss_rec, columns=['epoch', 'loss'])
            df_loss_log = df_loss_log.set_index('epoch')
            l_batch_loss_rec.append((batch_i, df_loss_log))
    df_batch_loss_log = pd.DataFrame(l_batch_loss_rec, columns=['batch', 'df_loss_log'])
    df_batch_loss_log = df_batch_loss_log.set_index('batch')
    print('[train_model] Training is done in %s secs.' % str(time.time() - timer_start))

    # SAVE MODELS
    time_stamp = time.strftime('%y%m%d%H%M%S', time.localtime())
    for model_idx, model_info in enumerate(l_top_models):
        model_info_dict = {
                            'loss': model_info[0],
                            'epoch': model_info[3],
                            'model_state_dict': model_info[1],
                            'optimizer_state_dict': model_info[2],
                            'train_hyperparams': g_train_hyperparams,
                            'model_global_hyperparams': g_model_hyperparams['global'],
                            'model_specific_hyperparams': g_model_hyperparams[model_name],
                            'loss_func_hyperparams': g_loss_func_hyperparams[loss_func_name],
                            'optimizer_hyperparams': g_optimizer_hyperparams[optimizer_name],
                            'batch_loss_log': df_batch_loss_log
                           }
        save_name = 'model#%s#%s#%s#%.2f#%s#%s#%s#%s.pickle' \
                    % (model_name, optimizer_name, loss_func_name, model_info[0], model_info[3], train_device,
                       time_stamp, model_idx)
        torch.save(model_info_dict, path.join(g_model_folder, save_name))
    print('[train_model] All done in %s secs.' % str(time.time() - timer_init))


def eval_model(model_info_name):
    """
    Evaluate performance of a model specified by 'model_info_name'.
    :param
        model_info_name: str
            The name of a saved model. See 'train_model()'.
    :return:

    """
    timer_init = time.time()
    print('[eval_model] Starts.')
    global g_eval_hyperparams
    if g_eval_hyperparams is None:
        raise Exception('[eval_model] g_eval_hyperparams is None.')
    saved_model_path = path.join(g_model_folder, model_info_name)
    if not path.exists(saved_model_path):
        raise Exception('[eval_model] %s does not exist.' % saved_model_path)

    # LOAD HYPERPARAMETERS
    model_info = torch.load(path.join(g_model_folder, model_info_name))
    train_hyperparams = model_info['train_hyperparams']
    model_name = train_hyperparams['model_name']
    eval_func_name = g_eval_hyperparams['eval_func_name']
    eval_set_name = g_eval_hyperparams['eval_set_name']
    eval_device = g_eval_hyperparams['eval_device']

    # LOAD EVALUATION DATA
    np_input, np_label = load_input_label_data(eval_set_name)

    # RECONSTRUCT LEARNED MODEL
    model = build_model(model_name)
    model.load_state_dict(model_info['model_state_dict'])
    model.eval()

    # CREATE EVALUATION FUNCTION
    eval_func = build_eval_func(eval_func_name)

    # EVALUATION
    timer_start = time.time()
    l_eval_score = []
    with torch.cuda.device(eval_device):
        for batch_i in range(np_input.shape[0]):
            th_batch_input = torch.from_numpy(np_input[batch_i]).type(torch.float32)
            th_batch_label = torch.from_numpy(np_label[batch_i]).type(torch.float32)
            th_batch_prediction = model(th_batch_input)
            eval_score = eval_func(th_batch_prediction, th_batch_label)
            if eval_score.shape == torch.Size([]):
                l_eval_score.append((batch_i, eval_score.item()))
            else:
                l_eval_score.append((batch_i, eval_score.numpy()))
            print('[eval_model] Batch %s: eval_score=%s, elapse=%s secs'
                  % (batch_i, eval_score, time.time() - timer_start))

    df_eval_score = pd.DataFrame(l_eval_score, columns=['batch', 'eval_score'])
    df_eval_score = df_eval_score.set_index('batch')
    pd.to_pickle(df_eval_score, path.join(g_results_folder, model_info_name[:-7]))

    print('[eval_model] All done in %s secs.' % str(time.time() - timer_init))
########################################################################################################################


########################################################################################################################
#   MODEL ZOO
########################################################################################################################
"""
Model Input:
    Shape: (num_seq, num_time, num_in_feature)

Model Output:
    Shape: (num_seq, num_time, num_out_feature)
"""


class CustomLSTM(nn.Module):
    """
    The input to 'm_lstm' should be a tensor of the shape (num_seq, num_time, num_in_feature) where 'num_seq' is
    the number of sequences, 'num_time' is the number of time points in each sequence, and 'num_in_feature' is the
    number of features describing a time point in the input. Note that these terms are different from the official
    PyTorch document of LSTM (https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#torch.nn.LSTM).
    Specifically, 'sequence' in our language means 'batch' in the PyTorch document, 'time point' in our language means
    'sequence' in the PyTorch document, and 'feature' in our language is referred as 'input' in the PyTorch document.

    the output from 'm_lstm' consists of three parts. 'last_layer_hiddens' contains the hidden state at each time point
    at the last LSTM layer only, and it is of the shape (num_seq, num_time, lstm_hidden_size), where 'lstm_hidden_size'
    is a hyperparameter of 'm_lstm'. 'final_hidden' contains the hidden states at the last time point in all
    LSTM layers, and it is of the shape (lstm_num_layers, num_seq, lstm_hidden_size). Similarly, 'final_cell' contains
    the cell states at the last time point in all LSTM layers, and it is of the shape
    (lstm_num_layers, num_seq, lstm_hidden_size).

    The input to 'm_dense_out' is of the shape (num_seq, lstm_hidden_size), and the output from 'm_dense_out' is of the
    shape (num_seq, 1, num_out_features).
    Note that when using RNN the prediction should only be 1 time point. It does not make sense that the prediction
    covers more than 1 time points, because each predicted time point needs to take the history right preceding the
    time point to be predicted.

    Note that the PyTorch LSTM implementation requires an initial hidden state and an initial cell state for each
    input sequence instead of sharing the initial states over all input sequences. Also, it is non-trivial to keep in
    mind that all input sequences are processed in a pure parallel way, i.e. no one affects any others.
    """
    def __init__(self, num_in_feature, num_out_features, out_seq_size, lstm_hidden_size, lstm_num_layers, lstm_dropout):
        super(CustomLSTM, self).__init__()
        self.m_lstm = nn.LSTM(input_size=num_in_feature,
                              hidden_size=lstm_hidden_size,
                              num_layers=lstm_num_layers,
                              batch_first=True,
                              dropout=lstm_dropout,
                              bidirectional=False)
        self.m_dense_out = nn.Linear(in_features=lstm_hidden_size,
                                     out_features=num_out_features)
        self.m_out_seq_size = out_seq_size

    def forward(self, input):
        last_layer_hiddens, (final_hidden, final_cell) = self.m_lstm(input)
        # TODO
        # Note that when computing the final output it can take into account more than 'final_hidden'.
        # For example, an Attention mechanism can be utilized over 'last_layer_hiddens', i.e. all hidden states
        # at the last layer, which potentially takes all hidden states into the computation. The intuition behind this
        # idea is that the predicted time point may be affected by some yet not all of the history time points (in the
        # input) more significantly than others.

        # Take the final hidden states at the last layer.
        output = self.m_dense_out(final_hidden[-1])
        output = output.reshape((output.shape[0], self.m_out_seq_size, output.shape[1]))
        return output


class NSEloss(nn.modules.loss._Loss):
    """
    As the shape of output from model is (num_seq, num_time, num_out_feature), the shape of input should be the same
    except that 'num_seq' and 'num_time' can be arbitrary. NSE is computed as follows:
        Generalized NSE = 1 - ( SUM[t:1~T](d(M_t,O_t)) / SUM[t:1~T](d(O_t,AVG[t:1~T](O_t))) )
    where 'M_t' is the model predicted value (which can be a vector) at time point 't', 'O_t' is the observation at
    time point 't', '1~T' represents the considered time points from 1 to T, 'd(*,*)' is a metric between two values
    at a time point, 'SUM[t](*)' represents the sum of values over t, and 'AVG[t](*)' represent the average of values
    over t.
    NSE takes values in [-inf, 1]. The higher the better. To offset issues caused by '-inf', a normalized NSE (NNSE) is
    computed:
        NNSE = 1 / (2 - NSE)
    NNSE takes values in [0, 1]. NNSE=1 correspnds to NSE=1, NNSE=0.5 corresponds to NSE=0, and NNSE=0 corresponds to
    NSE=-inf.

    Output:
        A single value is output if 'reduction' is specified. A vector otherwise.
    """
    def __init__(self, size_average=None, reduce=None, reduction='mean', time_point_metric='l2_sqr', normalized=False):
        """
        :param
            reduction: str (default='mean')
            - 'mean': Compute the mean of NSEs over all sequences.
            - 'sum': Compute the sum of NSEs over all sequences.
            - None: Output the vector of NSEs of all sequences without aggregation.
        :param
            time_point_metric: str (default='l2_sqr')
                The metric between input and target at one time point.
                - 'l2_sqr': The square of l2.
        :param:
            normalized: bool (default=False)
                - True: Use normalized NSE
                - False: Use unnormalized NSE
        """
        super(NSEloss, self).__init__(size_average, reduce, reduction)
        self.m_time_point_metric = time_point_metric
        self.m_normalized = normalized

    def forward(self, prediction, observation):
        """
        'prediction' and 'observation' should be of the same shape (num_seq, num_time, num_out_feature).
        :param
            prediction: PyTorch Tensor
        :param
            observation: PyTorch Tensor
        :return:
        """
        numerator_vec = None
        denominator_vec = None
        if self.m_time_point_metric == 'l2_sqr':
            mean_observation = torch.mean(observation, dim=1).reshape(observation.shape[0], 1, observation.shape[2])
            numerator_vec = torch.sum(torch.square(prediction - observation), dim=2)
            denominator_vec = torch.sum(torch.square(observation - mean_observation), dim=2)

        if numerator_vec is None or denominator_vec is None:
            eval_loss = None
        else:
            eval_loss = 1.0 - torch.sum(numerator_vec, dim=1) / torch.sum(denominator_vec, dim=1)
            if self.m_normalized:
                eval_loss = torch.div(1, 2 - eval_loss)

        return eval_loss
########################################################################################################################


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    plt.set_loglevel(level='error')

    cmd = sys.argv[1]

    if cmd == 'purge_task':
        print('[main:purge_task] purge_task starts.')
        purge_task()
        print('[main:purge_task] purge_task done.')

    elif cmd == 'preprocess':
        print('[main:preprocess] preprocess_data starts.')
        try:
            load_settings(['g_data_settings'])
            preprocess_data(g_task_name)
            split_data()
            build_input_and_label_data_for_train_val_test()
        except Exception as e:
            logging.error('[main:preprocess] %s' % e)
        print('[main:preprocess] preprocess_data done.')

    elif cmd == 'train':
        print('[main:train] train_model starts.')
        try:
            load_settings(['g_train_hyperparams', 'g_loss_func_hyperparams', 'g_optimizer_hyperparams',
                           'g_model_hyperparams'])
            train_model()
        except Exception as e:
            logging.error('[main:train] %s' % e)
        print('[main:train] train_model done.')

    elif cmd == 'eval':
        print('[main:eval] eval_model starts.')
        try:
            load_settings(['g_model_hyperparams', 'g_eval_hyperparams', 'g_eval_func_hyperparams'])
            model_info_name = 'model#custom_lstm#adagrad#mse#0.00#100#cuda#211210230729#1.pickle'
            eval_model(model_info_name)
        except Exception as e:
            logging.error('[main:eval] %s' % e)
        print('[main:eval] eval_model done.')
