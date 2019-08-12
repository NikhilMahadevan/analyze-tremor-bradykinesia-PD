'''
This file contains the code for filtering out predictions from each classifier based on context. The tree is as
follows:

  hand movement classifier
             |
    -----------------
    |YES            |NO
gait classifier   tremor classifier
    |NO             |YES
bradykinesia      tremor
assessment        assessment
'''
import pandas as pd

def filter_predictions_by_tree(algorithm_predictions):
    '''
    Filter out predictions based on context.

    :param algorithm_predictions: Pandas DataFrame with following columns = ['hand_movement', 'gait', 'tremor_constancy', 'tremor_amplitude', 'hand_movement_amplitude', 'hand_movement_jerk']
    :return: Pandas DataFrame of filtered predictions based on context.
    '''
    t_c_filtered = []
    t_a_filtered = []
    b_a_filtered = []
    b_j_filtered = []
    h_m_filtered = []

    for row in algorithm_predictions.itertuples():
        hm_p = row.hand_movement
        gait_p = row.gait
        trem_c_p = row.tremor_constancy
        trem_a_p = row.tremor_amplitude
        brady_amp_p = row.hand_movement_amplitude
        brady_j_p = row.hand_movement_jerk

        if hm_p == 0:
            b_a_filtered.append('NA')
            b_j_filtered.append('NA')
            h_m_filtered.append(hm_p)
            t_c_filtered.append(trem_c_p)
            if trem_c_p == 1:
                t_a_filtered.append(trem_a_p)
            else:
                t_a_filtered.append(0)
        else:
            t_c_filtered.append('NA')
            t_a_filtered.append('NA')
            if gait_p == 0:
                b_a_filtered.append(brady_amp_p)
                h_m_filtered.append(hm_p)
                b_j_filtered.append(brady_j_p)
            else:
                b_a_filtered.append('NA')
                h_m_filtered.append('NA')
                b_j_filtered.append('NA')

    final_data = pd.DataFrame()
    final_data['tremor_classifier_predictions'] = pd.Series(t_c_filtered)
    final_data['tremor_amplitude_predictions'] = pd.Series(t_a_filtered)
    final_data['hand_movement_predictions'] = pd.Series(h_m_filtered)
    final_data['hand_movement_amplitude'] = pd.Series(b_a_filtered)
    final_data['hand_movement_jerk'] = pd.Series(b_j_filtered)

    return final_data