'''
This file is used to extract features for gait classification. Machine learning model parameters are included.
Users will have to provide their own data and ground truths to train the model. Input data is raw accelerometer data
from wearable sensor on wrist location.
'''

import pandas as pd
from signal_preprocessing import preprocess
from features import signal_features as sf
import constants

def extract_gait_classification_features(window_data_df, channels, fs):
    '''
    Extract signal features applicable for gait classification for a given 3 second window of raw accelerometer data.

    :param window_data_df: Pandas DataFrame with columns ['ts','x','y','z']
    :param channels: Desired channels to run features on (Ex: ['x','y','z'])
    :param fs: Sampling rate of raw accelerometer data (Float)
    :return: DataFrame of calculated features on 3 second windows for given raw data
    '''
    features = pd.DataFrame()

    # Compute signal entropy
    feat_df_signal_entropy = sf.signal_entropy(window_data_df, channels)

    # Compute correlation coefficient
    feat_df_corr_coef = sf.correlation_coefficient(window_data_df, [['x_bp_filt_[0.25, 3.0]', 'y_bp_filt_[0.25, 3.0]'],
                                                                    ['x_bp_filt_[0.25, 3.0]', 'z_bp_filt_[0.25, 3.0]'],
                                                                    ['y_bp_filt_[0.25, 3.0]', 'z_bp_filt_[0.25, 3.0]']])

    # Compute RMS
    feat_df_signal_rms = sf.signal_rms(window_data_df, channels)

    # Compute range
    feat_df_signal_range = sf.signal_range(window_data_df, channels)

    # Compute IQR of Autocovariance
    feat_df_iqr_auto = sf.iqr_of_autocovariance(window_data_df, channels)

    # Compute Dominant Frequency
    sampling_rate = fs
    frequncy_cutoff = 12.0
    feat_df_dom_freq = sf.dominant_frequency(window_data_df, sampling_rate, frequncy_cutoff, channels)

    # Compute mean cross rate
    feat_df_mean_cross_rate = sf.mean_cross_rate(window_data_df, channels)

    # Compute range count percentage
    feat_df_range_count_percentage = sf.range_count_percentage(window_data_df, channels, min_value=-0.1, max_value=0.1)

    features = features.join(feat_df_signal_entropy, how='outer')
    features = features.join(feat_df_corr_coef, how='outer')
    features = features.join(feat_df_signal_rms, how='outer')
    features = features.join(feat_df_signal_range, how='outer')
    features = features.join(feat_df_iqr_auto, how='outer')
    features = features.join(feat_df_dom_freq, how='outer')
    features = features.join(feat_df_mean_cross_rate, how='outer')
    features = features.join(feat_df_range_count_percentage, how='outer')

    return features

def build_gait_classification_feature_set(raw_accelerometer_data_df, fs):
    '''
    Pre-process raw accelerometer data and compute signal based features on data.

    :param raw_accelerometer_data_df: Raw accelerometer data in a Pandas DataFrame wth columns = ['ts','x','y','z']
    :param fs: Sampling rate of raw accelerometer data (Float)
    :return: Pandas DataFrame of calculated features for given raw accelerometer data
    '''
    # Initialize final DataFrame
    final_feature_cache = pd.DataFrame()

    # Pre-process data
    # Bandpass filter between 0.25-3hz
    filtered_data_df = preprocess.band_pass_filter(raw_accelerometer_data_df, fs, [0.25, 3.0], 1,
                                                             channels=['x', 'y', 'z'])
    bp_headers = ['x_bp_filt_[0.25, 3.0]', 'y_bp_filt_[0.25, 3.0]', 'z_bp_filt_[0.25, 3.0]']

    # Perform PCA get 1st principal component for [0.25 - 3] bandpass filtered data
    filtered_data_df = preprocess.get_principal_component(filtered_data_df, channels=bp_headers)
    filtered_data_df.rename(columns={'PC1': 'PC1_[0.25, 3.0]'}, inplace=True)
    pca_headers = ['PC1_[0.25, 3.0]']

    total_data_channels = bp_headers + pca_headers

    # Segment into 3 second windows
    total_samples = filtered_data_df.shape[0]
    window_samples = fs * 3.0
    total_windows = round(total_samples / float(window_samples))

    for win in range(int(total_windows)):
        # Isolate data into windows
        current_win_start = int(window_samples * win + 1)
        current_win_end = int(current_win_start + window_samples - 1)
        if current_win_end > filtered_data_df.shape[0]:
            window_data_df = filtered_data_df.loc[current_win_start:, :]
        else:
            window_data_df = filtered_data_df.loc[current_win_start:current_win_end, :]
        window_data_df.reset_index(drop=True, inplace=True)

        # Extract Bradykinesia Features
        features_df = extract_gait_classification_features(window_data_df, total_data_channels, fs)

        # Discard window if NaN's in feature matrix
        if features_df.isnull().values.any():
            continue

        # Aggregate features for each window
        final_feature_cache = final_feature_cache.append(features_df, ignore_index=True)

    return final_feature_cache

def initialize_model():
    '''
    Function to create model that can be trained to classify gait using calculated signal based features.
    :return: SciKit learn Random Forest classifier
    '''
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()
    return model

if __name__ == "__main__":
    '''
    Main runner to extract features for gait classification. Model parameters are available to load (initialize_model()).  
    '''

    raw_data_filepath = ''# Insert filepath to raw accelerometer sensor data from wrist location.
    raw_data_df = pd.read_csv(raw_data_filepath) # Load data into Pandas DataFrame. (Channels = 'ts','x','y','z')

    fs = 128. # Enter sampling rate of raw accelerometer data

    # Build Feature Set
    feature_set = build_gait_classification_feature_set(raw_data_df, fs)

    # Filter out feature's based on feature selection
    feature_set = feature_set[constants.GAIT_FEATURE_SELECTION]

    # Load Model Configuration
    model = initialize_model()