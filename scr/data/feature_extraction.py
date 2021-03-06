import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, entropy


def feature_extraction(array:list, fs:int=50000):
    """feature_extraction: Summary

    Args:
        array (list):windowred data list
        fs (int): Sample Frequency (50 kHz)
    Returns:
        feature_dataframe: Extracted Features for all windowred data
    """
    feature_dataframe = pd.DataFrame(
          [[np.sqrt(np.mean(window_data ** 2)), kurtosis(window_data), entropy(probability(window_data)),
           fft_features(window_data, fs)] for window_data in array])
    spectral_data = feature_dataframe[3]
    feature_dataframe.drop(labels=[3], axis=1, inplace=True)
    feature_dataframe[[3, 4, 5, 6]] = pd.DataFrame(spectral_data.tolist(), index=feature_dataframe.index)
    return feature_dataframe

def probability(sample):
    dist = norm(np.mean(sample), np.std(sample))
    return dist.pdf(sample)

def fft_features(signal, f_sample):
    n = len(signal) // 2 + 1
    fft = 2.0 * np.abs(np.fft.fft(signal)[:n]) / n
    freq = np.linspace(0, f_sample / 2, n, endpoint=True)
    mean = np.mean(fft)
    centroid = np.average(fft, weights=freq)
    max_val = np.max(fft)
    kurt = kurtosis(fft)
    return mean, centroid, max_val, kurt

def feature_dataset(dataframe, fs, window_size, window_step,):
    last_index = len(dataframe) - 1
    df_appender = []
    for channel in (dataframe.columns.values.tolist()):
        #windowed_data = extract_windows_vectorized(dataframe[channel].to_numpy(), 0, last_index - window_step, window_size, window_step)
        windowed_data = rolling_window(dataframe[channel].to_numpy(), window_size, window_step)
        feature_df = feature_extraction(windowed_data, fs)
        feature_df.rename({0: 'RMS ' + channel, 1: 'Kurtosis ' + channel, 2: 'Entropy ' + channel,
                           3: 'Spectral Mean ' + channel, 4: 'Spectral Centroid ' + channel,
                           5: 'Spectral Maximum ' + channel, 6: 'Spectral Kurtosis ' + channel},
                          axis=1, inplace=True)
        df_appender.append(feature_df)
        print('finish feature extraction')
    #print(df_appender)
    return pd.concat(df_appender, axis=1)


def extract_windows_vectorized(array, clearing_time_index, max_time, sub_window_size, step_size):
    # max_time must be lower than (total lenght - step)
    # start = clearing_time_index + 1 - sub_window_size + 1
    start = clearing_time_index
    sub_windows = (
                    start +
                    np.expand_dims(np.arange(sub_window_size), 0) +
                    np.expand_dims(np.arange(max_time + 1, step=step_size), 0).T
                    )
    return array[sub_windows]

def rolling_window(array, window_size,window_step):
    shape = (array.shape[0] - window_size + 1, window_size)
    strides = (array.strides[0],) + array.strides
    rolled = np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)
    return rolled[np.arange(0,shape[0],window_step)]
