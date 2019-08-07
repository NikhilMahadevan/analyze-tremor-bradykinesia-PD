'''
This file contains code to characterize detected hand movement. These features are used to characterize bradykinesia.
The features calculated are: 1. hand movement amplitude, 2. smoothness of hand movement (jerk measure)
'''
import pandas as pd
import numpy as np
from signal_preprocessing import preprocess
from features import signal_features as sf

def compute_rms(data_array):
    '''
    Compute RMS of data.

    :param data_array: np.array of data
    :return: rms
    '''
    return np.sqrt(np.mean(np.square(data_array)))

def calculate_amplitude_and_smoothness_features(raw_accelerometer_data_df, fs):
    '''
    Function to calculate hand movement amplitude and smoothness of hand movement (jerk metric) from accelerometer data
    collected from a wrist worn wearable device.

    :param raw_accelerometer_data_df: Pandas DataFrame of raw accelerometer data. Columns = ['ts', 'x', 'y', 'z']
    :param fs: Sampling rate of raw accelerometer data (Float)
    :return: Computed hand movement amplitude (list) and smoothness of hand movement (jerk metric) (list) in 3 second
    windows
    '''

    # Pre-process data
    # Bandpass filter between 0.25-3.5
    filtered_data_df = preprocess.band_pass_filter(raw_accelerometer_data_df, fs, [0.25, 3.5], 4,
                                        channels=['x', 'y', 'z'])
    bp_headers = ['x_bp_filt_[0.25, 3.5]', 'y_bp_filt_[0.25, 3.5]', 'z_bp_filt_[0.25, 3.5]']
    filtered_data_df = filtered_data_df[bp_headers]

    avg_acc_per_window = []
    jerk_per_window = []

    # Segment into 3 second windows
    total_samples = filtered_data_df.shape[0]
    window_samples = fs * 3.0
    total_windows = round(total_samples / float(window_samples))
    for win in range(int(total_windows)):
        current_win_start = int(window_samples * win + 1)
        current_win_end = int(current_win_start + window_samples - 1)
        if current_win_end > filtered_data_df.shape[0]:
            window_data_df = filtered_data_df.loc[current_win_start:, :]
        else:
            window_data_df = filtered_data_df.loc[current_win_start:current_win_end, :]
        window_data_df.reset_index(drop=True, inplace=True)
        window_data_df = window_data_df[bp_headers]

        # Compute Avg RMS -> Amplitude of hand movement
        window_data_df['mag'] = np.sqrt(
            (window_data_df['x_bp_filt_[0.25, 3.5]'] ** 2) + (window_data_df['y_bp_filt_[0.25, 3.5]'] ** 2) + (
            window_data_df['z_bp_filt_[0.25, 3.5]'] ** 2))
        combined_amplitude = compute_rms(window_data_df.mag.tolist())
        avg_acc_per_window.append(combined_amplitude)

        # Compute Jerk -> Smoothness of hand movement
        computed_jerk_df = sf.jerk_metric(window_data_df, fs, ['mag'])
        computed_jerk = computed_jerk_df.mag_jerk_ratio.values[0]
        jerk_per_window.append(computed_jerk)

    return avg_acc_per_window, jerk_per_window


if __name__ == "__main__":
    '''
    Main runner to extract features of hand movement. 
    '''

    raw_data_filepath = ''# Insert filepath to raw accelerometer sensor data from wrist location.
    raw_data_df = pd.read_csv(raw_data_filepath) # Load data into Pandas DataFrame. (Channels = 'ts','x','y','z')

    fs = 128. # Enter sampling rate of raw accelerometer data

    # Build Feature Set
    amplitude, jerk = calculate_amplitude_and_smoothness_features(raw_data_df, fs)