'''
This file contains code to generate endpoints that characterize bradykinesia from data obtained from a wrist based
wearable sensor.

The endpoints are:
1. Amplitude of hand movement
2. Smoothness of hand movements
3. Percentage of time spent with no hand movements
4. Average length of bouts with no hand movement
'''
import numpy as np

def compute_aggregate_hand_movement_amplitude(hand_movement_amplitudes):
    '''
    Compute aggregate measures of hand movement amplitude.
    :param hand_movement_amplitudes: Computed hand movement amplitudes (list)
    :return: Average hand movement amplitude
    '''
    return np.mean(hand_movement_amplitudes)

def compute_aggregate_smoothness_of_hand_movement(hand_movement_jerk_predictions):
    '''
    Compute aggregate measures of smoothness of hand movement (jerk metric).
    :param hand_movement_jerk_predictions: Computed jerk metrics (list)
    :return: 95th percentile of jerk
    '''
    return np.percentile(hand_movement_jerk_predictions, 95)

def compute_aggregate_percentage_of_no_hand_movement(hand_movement_predictions):
    '''
    Compute aggregate value of percentage of no hand movement.
    :param hand_movement_predictions: Predicted hand movement - binary predictions (1 = hand movement, 0 = no hand movement).
    :return: Percentage of no hand movement
    '''
    return (hand_movement_predictions.count(0)/float(len(hand_movement_predictions)))*100.

def calculate_hand_movement_bout_lengths(data):
    '''
    Calculate bout lengths of no hand movement and hand movement
    :param data: Predicted hand movement - binary predictions (1 = hand movement, 0 = no hand movement).
    :return: bout lengths of no hand movement (list), bout lengths of hand movement (list)
    '''
    count_0 = 0
    count_1 = 0
    no_hand_movement_bouts = []
    hand_movement_bouts = []
    for idx, i in enumerate(data):
        if i == 0:
            count_1 = 0
            count_0+=1
            if idx+1<len(data):
                if data[idx+1]==1:
                    no_hand_movement_bouts.append(count_0)
        if i == 1:
            count_1+=1
            count_0 = 0
            if idx+1<len(data):
                if data[idx+1]==0:
                    hand_movement_bouts.append(count_1)

        if idx+1 == len(data):
            if count_0>0:
                no_hand_movement_bouts.append(count_0)
            if count_1>0:
                hand_movement_bouts.append(count_1)

    return no_hand_movement_bouts, hand_movement_bouts

def compute_aggregate_length_of_no_hand_movement_bouts(hand_movement_predictions):
    '''
    Compute aggregate length of no hand movement bouts
    :param hand_movement_predictions: Predicted hand movement - binary predictions (1 = hand movement, 0 = no hand movement).
    :return: Average length of no hand movement bout lengths
    '''
    no_hand_movement_bouts, _ = calculate_hand_movement_bout_lengths(hand_movement_predictions)
    return np.mean(no_hand_movement_bouts)

