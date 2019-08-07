'''
This file houses functions to compute signal features based on accelerometer data.
'''

from statsmodels.tsa.stattools import acf
from scipy import stats
import tsfresh as tsf
import numpy as np
import pandas as pd

def histogram(signal_x):
    '''
    Calculate histogram of sensor signal.

    :param signal_x: 1-D numpy array of sensor signal
    :return: Histogram bin values, descriptor
    '''
    descriptor = np.zeros(3)

    ncell = np.ceil(np.sqrt(len(signal_x)))

    max_val = np.nanmax(signal_x.values)
    min_val = np.nanmin(signal_x.values)

    delta = (max_val - min_val) / (len(signal_x) - 1)

    descriptor[0] = min_val - delta / 2
    descriptor[1] = max_val + delta / 2
    descriptor[2] = ncell

    h = np.histogram(signal_x, ncell.astype(int), range=(min_val, max_val))

    return h[0], descriptor

def signal_entropy(signal_df, channels):
    '''
    Calculate signal entropy of sensor signals.

    :param signal_df: Pandas DataFrame housing desired sensor signals
    :param channels: channels of signal to measure signal entropy
    :return: Pandas DataFrame housing calculated signal entropy for each signal channel
    '''
    signal_entropy_df = pd.DataFrame()

    for channel in channels:
        data_norm = signal_df[channel]/np.std(signal_df[channel])
        h, d = histogram(data_norm)

        lowerbound = d[0]
        upperbound = d[1]
        ncell = int(d[2])

        estimate = 0
        sigma = 0
        count = 0

        for n in range(ncell):
            if h[n] != 0:
                logf = np.log(h[n])
            else:
                logf = 0
            count = count + h[n]
            estimate = estimate - h[n] * logf
            sigma = sigma + h[n] * logf ** 2

        nbias = -(float(ncell) - 1) / (2 * count)

        estimate = estimate / count
        estimate = estimate + np.log(count) + np.log((upperbound - lowerbound) / ncell) - nbias

        # Scale the entropy estimate to stretch the range
        estimate = np.exp(estimate ** 2) - np.exp(0) - 1

        signal_entropy_df[channel + '_signal_entropy'] = [estimate]

    return signal_entropy_df

def correlation_coefficient(signal_df, channels):
    '''
    Calculate correlation coefficient of sensor signals.

    :param signal_df: Pandas DataFrame housing desired sensor signals
    :param channels: channels of signal to measure correlation coefficient
    :return: Pandas DataFrame of calculated correlation coefficient for each signal channel
    '''
    corr_coef_df = pd.DataFrame()
    C = signal_df.corr()

    for channel in channels:
        corr_coef_df[channel[0] + '_' + channel[1] + '_corr_coef'] = [C[channel[0]][channel[1]]]

    return corr_coef_df


def signal_rms(signal_df, channels):
    '''
    Calculate root mean square of sensor signals.

    :param signal_df: Pandas DataFrame housing desired sensor signals
    :param channels: channels of signal to measure RMS
    :return: Pandas DataFrame housing calculated RMS for each signal channel
    '''
    rms_df = pd.DataFrame()

    for channel in channels:
        rms_df[channel + '_rms'] = [np.std(signal_df[channel] - signal_df[channel].mean())]

    return rms_df

def signal_range(signal_df, channels):
    '''
    Calculate range of sensor signals.

    :param signal_df: Pandas DataFrame housing desired sensor signals
    :param channels: channels of signal to measure range
    :return: Pandas DataFrame housing calculated range for each signal channel
    '''
    range_df = pd.DataFrame()

    for channel in channels:
        range_df[channel + '_range'] = [signal_df[channel].max(skipna=True) - signal_df[channel].min(skipna=True)]

    return range_df

def iqr_of_autocovariance(signal_df, channels):
    '''
    Calculate interquartile range of autocovariance of sensor signals.
    
    :param signal_df: Pandas DataFrame housing sensor signals
    :param channels: channels of signal to obtain IQR of autocovariance
    :return: Pandas DataFrame of calculated IQR of autocovariance for each signal channel
    '''
    autocov_range_df = pd.DataFrame()

    n_samples = signal_df.shape[0]
    for channel in channels:
        current_autocov_iqr = stats.iqr(acf(signal_df[channel], unbiased=True, nlags=n_samples/2))
        autocov_range_df[channel + '_iqr_of_autocovariance'] = [current_autocov_iqr]

    return autocov_range_df

def dominant_frequency(signal_df, sampling_rate, cutoff, channels):
    '''
    Calculate dominant frequency of sensor signals.

    :param signal_df: Pandas DataFrame housing desired sensor signals
    :param sampling_rate: sampling rate of sensor signal
    :param cutoff: desired cutoff for filter
    :param channels: channels of signal to measure dominant frequency
    :return: Pandas DataFrame of calculated dominant frequency for each signal channel
    '''
    dominant_freq_df = pd.DataFrame()

    for channel in channels:
        signal_x = signal_df[channel]

        padfactor = 1
        dim = signal_x.shape
        nfft = 2 ** ((dim[0] * padfactor).bit_length())

        freq_hat = np.fft.fftfreq(nfft) * sampling_rate
        freq = freq_hat[0:nfft / 2]

        idx1 = freq <= cutoff
        idx_cutoff = np.argwhere(idx1)
        freq = freq[idx_cutoff]

        sp_hat = np.fft.fft(signal_x, nfft)
        sp = sp_hat[0:nfft / 2] * np.conjugate(sp_hat[0:nfft / 2])
        sp = sp[idx_cutoff]
        sp_norm = sp / sum(sp)

        max_freq = freq[sp_norm.argmax()][0]
        max_freq_val = sp_norm.max().real

        idx2 = (freq > max_freq - 0.5) * (freq < max_freq + 0.5)
        idx_freq_range = np.where(idx2)[0]
        dom_freq_ratio = sp_norm[idx_freq_range].real.sum()

        # Calculate spectral flatness
        spectral_flatness = 10.0*np.log10(stats.mstats.gmean(sp_norm)/np.mean(sp_norm))

        # Estimate spectral entropy
        spectral_entropy_estimate = 0
        for isess in range(len(sp_norm)):
            if sp_norm[isess] != 0:
                logps = np.log2(sp_norm[isess])
            else:
                logps = 0
            spectral_entropy_estimate = spectral_entropy_estimate - logps * sp_norm[isess]

        spectral_entropy_estimate = spectral_entropy_estimate / np.log2(len(sp_norm))
        # spectral_entropy_estimate = (spectral_entropy_estimate - 0.5) / (1.5 - spectral_entropy_estimate)

        dominant_freq_df[channel + '_dom_freq_value'] = [max_freq]
        dominant_freq_df[channel + '_dom_freq_magnitude'] = [max_freq_val]
        dominant_freq_df[channel + '_dom_freq_ratio'] = [dom_freq_ratio]
        dominant_freq_df[channel + '_spectral_flatness'] = [spectral_flatness[0].real]
        dominant_freq_df[channel + '_spectral_entropy'] = [spectral_entropy_estimate[0].real]

    return dominant_freq_df

def mean_cross_rate(signal_df, channels):
    '''
    Compute mean cross rate of sensor signals.

    :param signal_df: Pandas DataFrame housing desired sensor signals
    :param channels: channels of signal to measure mean cross rate
    :return: Pandas DataFrame housing calculated mean cross rate for each signal channel
    '''
    mean_cross_rate_df = pd.DataFrame()
    signal_df_mean = signal_df[channels] - signal_df[channels].mean()

    for channel in channels:
        MCR = 0

        for i in range(len(signal_df_mean) - 1):
            if np.sign(signal_df_mean.loc[i, channel]) != np.sign(signal_df_mean.loc[i + 1, channel]):
                MCR += 1

        MCR = float(MCR) / len(signal_df_mean)

        mean_cross_rate_df[channel + '_mean_cross_rate'] = [MCR]

    return mean_cross_rate_df

def range_count_percentage(signal_df, channels, min_value=-1, max_value=1):
    '''
    Calculate range count percentage of sensor signals.

    :param signal_df: Pandas DataFrame housing desired sensor signals
    :param channels: channels of signal to measure range count percentage
    :param min_value: desired minimum value
    :param max_value: desired maximum value
    :return: Pandas DataFrame of calculated range count percentage for each signal channel
    '''
    range_count_df = pd.DataFrame()

    for channel in channels:
        signal_x = signal_df[channel]
        current_range_count = tsf.feature_extraction.feature_calculators.range_count(signal_x, min_value, max_value) * 1.0 / len(signal_x)
        range_count_df[channel + '_range_count_per'] = [current_range_count]

    return range_count_df

def jerk_metric(signal_df, sampling_rate, channels):
    '''
    Calculate jerk of sensor signals.

    :param signal_df: Pandas DataFrame housing desired sensor signals
    :param sampling_rate: sampling rate of sensor signals
    :param channels: channels of sensor signal to compute jerk
    :return: Pandas DataFrame of calculated jerk for each sensor channel
    '''
    jerk_ratio_df = pd.DataFrame()

    dt = 1. / sampling_rate
    duration = len(signal_df) * dt

    for channel in channels:
        amplitude = max(abs(signal_df[channel]))

        jerk = signal_df[channel].diff(1) / dt
        jerk_squared = jerk ** 2
        jerk_squared_sum = jerk_squared.sum(axis=0)
        scale = 360 * amplitude ** 2 / duration

        mean_squared_jerk = jerk_squared_sum * dt / (duration * 2)

        jerk_ratio_df[channel + '_jerk_ratio'] = [mean_squared_jerk / scale]

    return jerk_ratio_df