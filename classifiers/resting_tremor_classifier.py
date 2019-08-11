'''
This file is used to extract features for resting tremor classification. Machine learning model parameters are included.
Users will have to provide their own data and ground truths to train the model. Input data is raw accelerometer data
from wearable sensor on wrist location.
'''
import pandas as pd
from signal_preprocessing import preprocess
from features import signal_features as sf
import constants

def extract_tremor_classification_features(data_df, current_feature_df, data_channels, fs):
    '''
    Compute signal features applicable for tremor classification for a given 3 second window.
    :param data_df: Raw accelerometer data as Pandas DataFrame. Columns = ['ts', 'x', 'y', 'z']
    :param current_feature_df: Pandas DataFrame of current features to append new features to.
    :param data_channels: Data channels to run features on. Ex: ['x','y','z']
    :param fs: Sampling rate of raw accelerometer data (float)
    :return: Pandas DataFrame of computed features for given window of data.
    '''
    # Compute signal range
    feat_df_range = sf.signal_range(data_df, channels=data_channels)
    current_feature_df = current_feature_df.join(feat_df_range, how='outer')

    # Compute RMS
    feat_df_rms = sf.signal_rms(data_df, channels=data_channels)
    current_feature_df = current_feature_df.join(feat_df_rms, how='outer')

    # Compute Dominant Frequency
    frequncy_cutoff = 12.0
    feat_df_dom_freq = sf.dominant_frequency(data_df, fs, frequncy_cutoff, data_channels)
    current_feature_df = current_feature_df.join(feat_df_dom_freq, how='outer')

    # Compute Entropy
    feat_df_entropy = sf.signal_entropy(data_df, channels=data_channels)
    current_feature_df = current_feature_df.join(feat_df_entropy, how='outer')

    return current_feature_df

def build_rest_tremor_classification_feature_set(raw_accelerometer_data_df, fs):
    '''
    Pre-process raw accelerometer data and compute signal based features on pre-processed signal data.

    :param raw_accelerometer_data_df: Pandas DataFrame of raw accelerometer data. Columns = ['ts','x','y','z']
    :param fs: Sampling rate of raw accelerometer data (float)
    :return: Pandas DataFrame of calculated features in 3 second windows.
    '''
    # Initialize final DataFrame
    final_feature_set = pd.DataFrame()

    # Pre-process data
    # Bandpass filter between 3.5-7.5 hz
    filtered_data_df = preprocess.band_pass_filter(raw_accelerometer_data_df, fs, [3.5, 7.5],
                                                             1, channels=['x', 'y', 'z'])
    bp1_headers = ['x_bp_filt_[3.5, 7.5]', 'y_bp_filt_[3.5, 7.5]', 'z_bp_filt_[3.5, 7.5]']

    # Bandpass filter between 0.25-3.5 hz
    filtered_data_df = preprocess.band_pass_filter(filtered_data_df, fs, [0.25, 3.5],
                                                             1, channels=['x', 'y', 'z'])
    bp2_headers = ['x_bp_filt_[0.25, 3.5]', 'y_bp_filt_[0.25, 3.5]', 'z_bp_filt_[0.25, 3.5]']

    # Perform PCA get 1st principal component for [3.5 - 7.5] bandpass filtered data
    filtered_data_df = preprocess.get_principal_component(filtered_data_df, channels=bp1_headers)
    filtered_data_df.rename(columns={'PC1': 'PC1_[3.5, 7.5]'}, inplace=True)
    pca1_headers = ['PC1_[3.5, 7.5]']

    # Perform PCA get 1st principal component for [0.25 - 3.5] bandpass filtered data
    filtered_data_df = preprocess.get_principal_component(filtered_data_df, channels=bp2_headers)
    filtered_data_df.rename(columns={'PC1': 'PC1_[0.25, 3.5]'}, inplace=True)
    pca2_headers = ['PC1_[0.25, 3.5]']

    # Obtain all data channels of interest
    total_data_channels = bp1_headers + bp2_headers + pca1_headers + pca2_headers

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

        # Create DataFrame for current window of data
        current_features_df = pd.DataFrame()
        # Extract signal features for tremor detection
        current_features_df = extract_tremor_classification_features(window_data_df, current_features_df, total_data_channels, fs)
        # Aggregate features for each window
        final_feature_set = final_feature_set.append(current_features_df, ignore_index=True)

    return final_feature_set

def initialize_model():
    '''
    Model that can be trained to classify periods of tremor using calculated signal based features.
    :return: SciKit Learn Random Forest classifier
    '''
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()
    return model

if __name__ == "__main__":
    '''
    Main runner to extract features for tremor classification. Model parameters are available to load.  
    '''

    raw_data_filepath = ''# Insert filepath to raw accelerometer sensor data from wrist location.
    raw_data_df = pd.read_csv(raw_data_filepath) # Load data into Pandas DataFrame. (Channels = 'ts','x','y','z')

    fs = 128. # Enter sampling rate of raw accelerometer data

    # Build Feature Set
    feature_set = build_rest_tremor_classification_feature_set(raw_data_df, fs)

    # Filter out feature's based on feature selection
    feature_set = feature_set[constants.TREMOR_FEATURE_SELECTION]

    # Load Model Configuration
    model = initialize_model()