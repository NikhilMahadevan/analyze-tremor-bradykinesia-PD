'''
This file contains code to run a hand movement classifier based on accelerometer data obtained from a wearable
device located on the wrist.
'''

import pandas as pd
import numpy as np
from scipy import signal

def compute_rolling_mean(x, window_length):
    '''
    Method to compute rolling mean.
    :param x: 1D numpy array
    :param window_length: Length of window for computing rolling mean. Must be an odd number.
    :return: Numpy array with rolling mean values calculated over given window length.
    '''
    if window_length % 2 == 0:
        print "Window length should be an odd number."
        return

    y = np.zeros(len(x))
    for i in range(len(x)):
        if i < window_length/2:
            y[i] = np.mean(x[0:i + window_length / 2])
        elif len(x) - i < window_length / 2:
            y[i] = np.mean(x[i - window_length / 2:])
        else:
            y[i] = np.mean(x[i - window_length / 2: i + window_length / 2])

    return y

def compute_rolling_std(x, window_length):
    '''
    Method to compute rolling standard deviation.
    :param x: 1D numpy array
    :param window_length: Length of window for computing rolling standard deviation. Must be an odd number.
    :return: Numpy array with rolling standard deviation values calculated over given window length.
    '''
    if window_length % 2 == 0:
        print "Window length should be an odd number."
        return

    y = np.zeros(len(x))
    for i in range(len(x)):
        if i < window_length/2:
            y[i] = np.std(x[0:i + window_length / 2])
        elif len(x) - i < window_length / 2:
            y[i] = np.std(x[i - window_length / 2:])
        else:
            y[i] = np.std(x[i - window_length / 2: i + window_length / 2])

    return y

def detect_hand_movement(raw_accelerometer_data_df, fs, window_length=3, threshold=0.01):
    '''
    Method for detecting hand movement from raw accelerometer data.
    :param raw_accelerometer_data_df: Pandas DataFrame with accelerometer axis represented as x, y and z columns
    :param fs: Sampling rate (samples/second) of the accelerometer data
    :param window_length: Length (in seconds) of the non-overlapping window for hand movement classification
    :param threshold: Threshold value that is applied to the coefficient of variation to detect hand movement
    :return: Detected hand movement as numpy array in desired window length
    '''
    # Calculate the vector magnitude of the accelerometer signal
    accelerometer_vector_magnitude = np.sqrt((raw_accelerometer_data_df.x**2 + raw_accelerometer_data_df.y**2 + raw_accelerometer_data_df.z**2))

    # Low-pass filter the accelerometer vector magnitude signal to remove high frequency components
    low_pass_cutoff = 3 # cutoff frequency for the lowpass filter
    wn = [low_pass_cutoff * 2 / fs]
    [b, a] = signal.iirfilter(6, wn, btype='lowpass', ftype = 'butter')
    accelerometer_vector_magnitude_filt = signal.filtfilt(b, a, accelerometer_vector_magnitude)

    # Calculate the rolling coefficient of variation
    rolling_mean = compute_rolling_mean(accelerometer_vector_magnitude_filt, int(fs+1))
    rolling_std = compute_rolling_std(accelerometer_vector_magnitude_filt, int(fs+1))
    rolling_cov = rolling_std/rolling_mean

    # Detect CoV values about given movement threshold
    values_above_threshold = (rolling_cov > threshold)*1

    # Classify non-overlapping windows as either hand movement or no hand movement
    samples_in_window = window_length * fs

    if len(rolling_cov) / samples_in_window > np.floor(len(rolling_cov) / samples_in_window):
        number_of_windows = int(round(len(rolling_cov) / samples_in_window))
    else:
        number_of_windows = int(np.floor(len(rolling_cov) / samples_in_window))

    window_labels = np.zeros(number_of_windows)
    for iwin in range(number_of_windows):

        if iwin == number_of_windows:
            win_start = iwin * samples_in_window
            win_stop = len(rolling_cov)
        else:
            win_start = iwin * samples_in_window
            win_stop = (iwin + 1) * samples_in_window

            if np.mean(values_above_threshold[int(win_start):int(win_stop)]) >= 0.5:
                window_labels[iwin] = 1

    return window_labels

if __name__ == "__main__":
    '''
    Main runner for hand movement detection from accelerometer data located at the wrist location. 
    '''

    raw_data_filepath = '' # Insert file path of raw accelerometer data

    wrist_accel = pd.read_csv(raw_data_filepath)
    sampling_rate = 128.0 # Specify sampling rate of sensor data
    window_length = 3 # Specify window length (in seconds) to output hand movement predictions

    # Run hand movement detection
    window_labels = detect_hand_movement(wrist_accel, window_length=window_length, fs=sampling_rate)