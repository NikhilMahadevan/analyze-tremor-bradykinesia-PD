'''
This file houses functions used for pre-processing accelerometer sensor signals.
'''

from scipy import signal
import pandas as pd
from sklearn.decomposition import PCA

def band_pass_filter(data_df, sampling_rate, bp_cutoff, order, channels=['X', 'Y', 'Z']):
    '''
    Band-pass filter a given sensor signal.

    :param data_df: dataframe housing sensor signals
    :param sampling_rate: sampling rate of signal
    :param bp_cutoff: filter cutoffs
    :param order: filter order
    :param channels: channels of signal to filter
    :return: dataframe of raw and filtered data
    '''
    data = data_df[channels].values

    # Calculate the critical frequency (radians/sample) based on cutoff frequency (Hz) and sampling rate (Hz)
    critical_frequency = [bp_cutoff[0]* 2.0 / sampling_rate, bp_cutoff[1]* 2.0 / sampling_rate]

    # Get the numerator (b) and denominator (a) of the IIR filter
    [b, a] = signal.butter(N=order, Wn=critical_frequency, btype='bandpass', analog=False)

    # Apply filter to raw data
    bp_filtered_data = signal.filtfilt(b, a, data, padlen=10, axis=0)

    new_channel_labels = [ax + '_bp_filt_' + str(bp_cutoff) for ax in channels]

    data_df[new_channel_labels] = pd.DataFrame(bp_filtered_data)

    return data_df

def get_principal_component(data_df, channels=['X', 'Y', 'Z'], n_components=1):
    '''
    Compute principal components of sensor signal.

    :param data_df: dataframe housing sensor signals
    :param channels: channels of sensor signal to compute principal component analysis on
    :return: dataframe of raw data and principal components
    '''
    pca = PCA(n_components=n_components, svd_solver='arpack')

    principal_component = pca.fit_transform(data_df[channels])

    principal_component_df = pd.DataFrame(principal_component)
    cols = ['PC'+str(i+1) for i in range(0,n_components)]
    principal_component_df.columns = cols

    data_df = pd.concat([data_df, principal_component_df], axis=1)

    return data_df