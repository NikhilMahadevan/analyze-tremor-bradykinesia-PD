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
    hm = algorithm_predictions.hand_movement.tolist()
    gait = algorithm_predictions.gait.tolist()
    tremor_constancy = algorithm_predictions.tremor_constancy.tolist()
    tremor_amp = algorithm_predictions.tremor_amplitude.tolist()
    brady_amp = algorithm_predictions.brady_amplitude.tolist()
    brady_jerk = algorithm_predictions.brady_jerk.tolist()

    t_c_filtered = []
    t_a_filtered = []
    b_a_filtered = []
    b_j_filtered = []
    h_m_filtered = []

    for hm_p, gait_p, trem_c_p, trem_a_p, brady_amp_p, brady_j_p in zip(hm, gait, tremor_constancy, tremor_amp, brady_amp, brady_jerk):
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
            if gait == 0:
                b_a_filtered.append(brady_amp_p)
                h_m_filtered.append(hm_p)
                b_j_filtered.append(brady_j_p)


    final_data = pd.DataFrame()
    final_data['tremor_classifier_predictions'] = pd.Series(t_c_filtered)
    final_data['tremor_amplitude_predictions'] = pd.Series(t_a_filtered)
    final_data['hand_movement_predictions'] = pd.Series(h_m_filtered)
    final_data['hand_movement_amplitude'] = pd.Series(b_a_filtered)
    final_data['hand_movement_jerk'] = pd.Series(b_j_filtered)

    return final_data