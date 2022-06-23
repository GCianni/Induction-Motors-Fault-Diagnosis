
import glob
import os
import pandas as pd
import numpy as np
import sys

from scipy.signal import butter, lfilter
from feature_extraction import feature_dataset

def get_datafiles(data_path:str, label_path:str)->list:
    """get_datafiles Summary: 
        The script open all files in path and appending them in a single list
    Args:
        data_path (str): dataset csv files path
        label_path(str): dataset label txt file path
    Returns:
        dataframe_list: list of all opened datasets file
    """
    all_files = glob.glob(os.path.join(data_path, "*.csv"))
    dataframe_list = [pd.read_csv(filename,
                                  names=["Acc_Inner_X", "Acc_Inner_Y", "Acc_Inner_Z",
                                         "Acc_Outer_X", "Acc_Outer_Y", "Acc_Outer_Z"],
                                  usecols=[1, 2, 3, 4, 5, 6])
                      for filename in all_files]

    data_label = get_data_label(label_path)
    return dataframe_list, data_label, all_files

def get_data_label(label_path:str)->list:
    """get_data_label Summary:
        The Script open the label.txt file as a list
    Args:
        label_path (str): dataset label txt file path
    Returns:
        label_list: Datafiles label list sorted by get_datafiles order 
    """
    target_df = pd.read_csv(label_path,
                            delimiter="\t", header=None)
    # print(target_df)
    label_list = target_df.values.tolist()
    return label_list[0]

def lowpass_filter(data, cutoff:int, fs:int, order:int=5):
    """lowpass_filter:Summary
    Args:
        data (_type_): _description_
        cutoff (_type_): _description_
        fs (_type_): _description_
        order (int, optional): _description_. Defaults to 5.
    Returns:
        _type_: _description_
    """
    normal_cutoff = cutoff / (0.5 * fs)
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = lfilter(b, a, data).tolist()
    return filtered_data

def filter_data(dataframes:list, cutoff_freq:int, fs:int)->list:
    filtered_dataframes = [pd.DataFrame(lowpass_filter(data, cutoff_freq, fs),
                                        columns=data.columns)
                        for data in dataframes]
    return filtered_dataframes

def get_featured_labeled_data(label, dataframe, fs:int, window_size:int, window_step:int, name):
    data = feature_dataset(dataframe=dataframe, fs=fs, window_size=window_size, window_step=window_step)
    data['Target'] = pd.Series(np.full(len(data.index), label))
    print(name)
    # data.to_csv('C:\\Monografia\\data\\saved_features_data\\'+name+'.csv') # Salva cada extração de característica para cada csv
    return data

