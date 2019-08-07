'''
This file contains code to generate endpoints from resting tremor classifiers. The endpoints are:
1. Tremor Constancy
2. Tremor Amplitude
'''
import numpy as np

def compute_tremor_constancy(tremor_classification_predictions):
    '''
    Compute tremor constancy for a given set of tremor predictions.
    :param tremor_classification_predictions: Tremor predictions as determined by tremor classifier. Binary predictions (1 = tremor, 0 = no tremor)
    :return: Percentage of detected tremor
    '''
    return tremor_classification_predictions.count(1)/float(len(tremor_classification_predictions))*100.

def compute_aggregate_tremor_amplitude(tremor_amplitude_predictions):
    '''
    Compute an aggregate measure of tremor amplitude for a given set of tremor amplitude predictions.
    :param tremor_amplitude_predictions: Computed tremor amplitude
    :return: 85th percentile of tremor amplitude predictions.
    '''
    return np.percentile(tremor_amplitude_predictions, 85)