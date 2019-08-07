'''
This file contains code to compute resting tremor amplitude from accelerometer data obtained from wearable sensors
located on the wrist.
'''

import pandas as pd
import numpy as np
from signal_preprocessing import preprocess

def compute_rms(data_array):
    '''
    Compute RMS of data.

    :param data_array: np.array of data
    :return: rms
    '''
    return np.sqrt(np.mean(np.square(data_array)))

def calculate_tremor_amplitude(raw_accelerometer_data_df, fs):
    '''
    Calculate tremor amplitude from raw accelerometer data collected from wearable sensor at wrist location.
    :param raw_accelerometer_data_df: Pandas DataFrame of raw accelerometer data. Columns = ['ts','x','y','z']
    :param fs: Sampling rate of raw accelerometer data (float)
    :return: Computed tremor amplitude in 3 second windows (list)
    '''

    # Pre-process data
    # Bandpass filter between 3.5-7.5
    filtered_data_df = preprocess.band_pass_filter(raw_accelerometer_data_df, fs, [3.5, 7.5], 3,
                                                             channels=['x', 'y', 'z'])
    bp_headers = ['x_bp_filt_[3.5, 7.5]', 'y_bp_filt_[3.5, 7.5]', 'z_bp_filt_[3.5, 7.5]']
    filtered_data_df = filtered_data_df[bp_headers]

    tremor_amplitudes_per_window = []

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

        # Compute Tremor Amplitude
        window_data_df['mag'] = np.sqrt((window_data_df['x_bp_filt_[3.5, 7.5]']**2)+(window_data_df['y_bp_filt_[3.5, 7.5]']**2)+(window_data_df['z_bp_filt_[3.5, 7.5]']**2))
        combined_amplitude = compute_rms(window_data_df.mag.tolist())
        tremor_amplitudes_per_window.append(combined_amplitude)

    return tremor_amplitudes_per_window

if __name__ == "__main__":
    '''
    Main runner for resting tremor amplitude computation from accelerometer data located at the wrist location. 
    '''

    raw_data_filepath = '' # Insert file path of raw accelerometer data

    raw_data_df = pd.read_csv(raw_data_filepath)
    sampling_rate = 128.0 # Specify sampling rate of sensor data

    # Run tremor amplitude calculation
    calculate_tremor_amplitude(raw_data_df, sampling_rate)