# analyze-tremor-bradykinesia-PD
A Python (2.7) package that enables digital measurements of **resting tremor** and **bradykinesia** in patients with Parkinson's Disease. These measurements utilize accelerometer data as input from a single wrist-worn wearable device located on the most affected side.

## Overview
Objective assessment of Parkinson’s disease symptoms during free-living conditions can provide valuable information for disease management and help accelerate the development of new therapies. Traditional assessments are episodic (require patients to come into a clinic at specified intervals for assessment), low-fidelity (assessments are paper-based rudimentary numerical rating scales), and subjective in nature (assessments are reported by clinicians and patients). Recent advances in wearable sensor technology have enabled the use of objective digital measurements to assess Parkinson's disease symptoms in free-living conditions. Current digital measurements often require the use of multiple devices or performance of prescribed motor activites, which is not optimal for monitoring free-living conditions.

Herein we present our source code used for the development and validation of a method aimed at objective assessment of **resting tremor** and **bradykinesia** (two common symptoms of Parkinson's disease) using accelerometer data captured with a single wrist-worn device during the performance of unscripted activities. Our method combines context detection and symptom assessment by using heuristic and machine learning models in a hierarchical framework to provide continuous monitoring by sequentially processing epochs of raw sensor data. Results of our analysis show that sensor derived continuous measures of resting tremor and bradykinesia achieve good to strong agreement with clinical assessment of symptom severity and are able to discriminate between treatment related changes in Parkinsonian motor states (ON/OFF).

## Software Requirements
There are 7 main packages used in this repository. The names of the packages and versions are listed below:

* ``pandas``: 0.23.4+
* ``scipy``: 1.2.2+
* ``scikit-learn``: 0.20.0+
* ``statsmodels``: 0.8.0+
* ``tsfresh``: 0.11.0
* ``numpy``: 1.16.4+

If necessary, the listed requirements can be installed as follows:
```
pip install -r requirements.txt
```

## Repository Contents
Our method for continuous objective of assessment of resting tremor and bradykinesia follows the hierarchical framework seen below:

<p align="center">
  <img width="500" height="450" src="https://raw.githubusercontent.com/NikhilMahadevan/analyze-tremor-bradykinesia-PD/master/images/pd_analytics_diagram.png?token=ABFEV6RQMG7HO6VN6ZWSRIS5L7YKY">
</p>

This system utilizes heuristic and machine learning models for context detection and symptom assessment. This repository contains the source code for each module. Currently the data set used to support the findings of this work is restricted for public use. All heuristic models are available, but for machine learning models only the code for generating the signal based features used as input into model training and model parameters are available. Users of this source code will have to provide their own labeled data sets for training each of the machine learning models.

The repository is organized as follows:
* __classifiers__: code to generate classifiers in each node of the tree above. See further explanation in table below:

|File | Model Type | Description |
| --- | --- | --- |
| hand_movement_classifier.py | Heuristic | Binary classification of hand movement |
| resting_tremor_classifier.py | Machine Learning | Binary classification of resting tremor |
| gait_classifier.py | Machine Learning | Binary classification of gait |
| resting_tremor_amplitude.py | Heuristic | Compute tremor amplitude |
| hand_movement_features.py | Heuristic | Compute amplitude of hand movement and smoothness of hand movement (jerk metric) |

* __endpoints__: code to filter model predictions per the tree above and summarize measures of resting tremor and bradykinesia for a given period of time. See further explanation in table below:

|File| Description|
|---|---|
| filter_classifier_predictions.py | Filter model predictions per tree above |
| bradykinesia_endpoints.py | Calculate: <ul><li>Mean bouts of no hand movement</li><li>Percentage of no hand movement</li><li>Mean hand movement amplitude</li><li>95th percentile of smoothness of hand movement</li></ul> |
| resting_tremor_endpoints.py | Calculate: <ul><li>Percentage of tremor (tremor constancy)</li><li>85th percentile of tremor amplitude</li></ul> |

* __signal_preprocessing__: signal preprocessing functions applied on accelerometer data prior to feature extraction
* __features__: signal features extracted from accelerometer data used to train supervised learning machine learning models

## Demo
A demo utilizing each of the functions explained above can be seen in the iPython notebook `demo_run_analytics.ipynb` in the `demo` folder. Since there are restrictions on the data set used with our work, the example data used for the demo is not from a Parkinson's patient and should not be used to analyze symptom endpoints. The demo is purely used to show how to make use of the code. Please see below section **Instructions for Use** for a more detailed explanation.

The iPython notebook server can be started via the command line by navigating to the root folder of this code base and then typing: `jupyter notebook`. Once the server is started, navigate to the `demo` folder and open the `demo_run_analytics.ipynb` file.

## Instructions for Use
Each classifer file is set up to be run independently but can also chained together in a higher level function. Each file requires a filepath to raw accelerometer data (as a **.CSV file (unit = G's)**) from a wrist-worn device and the sampling rate of the accelerometer data. The raw accelerometer .CSV file should be organized with the following column headers: `'ts','x','y','z'`. The code is set up to be manipulated by the user. Each file has a `if __name__ == "__main__":` section which serves as the entry point for all relevant functions in each file. All processing code in this repository is set up to run on **3 second windows**.

An example of how to make use of this code is as follows:

1. First train, validate, and test a gait and tremor machine learning classifier. We used a leave-one-subject-out routine to train and validate each machine learning classifier. The `classifiers/resting_tremor_classifier.py` and `classifiers/gait_classifier.py` files enable signal pre-processing, feature extraction, and feature selection for each classifier respectively (as used in our work). Also the default parameters of each machine learning model used can be loaded (untrained) via: `model = initialize_model()` function call. The user can then train the respective model via the `scikit-learn` `.fit` function (ex: `model.fit(feature_set`)).

2. Next, the user can run the raw accelerometer data through `classifiers/hand_movement_classifier.py` to get binary hand movement predictions, `classifiers/resting_tremor_amplitude_classifier.py` to get resting tremor amplitude predictions, and `classifiers/hand_movement_features.py` to get predictions for hand movement amplitude and smoothness of hand movement (jerk metric).

3. The user can then organize all the predictions from each module into a single `Pandas DataFrame` with the following column headers: `'hand_movement', 'gait', 'tremor_constancy', 'tremor_amplitude', 'hand_movement_amplitude', 'hand_movement_jerk'`. These predictions can then be filtered using the hierarchical tree shown above with the function: `filter_predictions_by_tree()` in the `endpoints/filter_classifier_predictions.py` file. This function will return another `Pandas DataFrame` of predictions with the following columns: `'tremor_classifier_predictions','tremor_amplitude_predictions','hand_movement_predictions', 'hand_movement_amplitude', 'hand_movement_jerk'`.

4. Summary measures can be generated for each measurement. Summary measures of tremor (tremor constancy and tremor amplitude) can be computed using the functions `compute_tremor_constancy()` and `compute_aggregate_tremor_amplitude()` respectively from the file `endpoints/resting_tremor_endpoints.py`. Summary measures of bradykinesia (hand movement amplitude, smoothness of hand movement, percentage of no hand movement, length of no hand movement bouts) can be computed using `compute_aggregate_hand_movement_amplitude()`, `compute_aggregate_smoothness_of_hand_movement()`, `compute_aggregate_percentage_of_no_hand_movement()`, `compute_aggregate_length_of_no_hand_movement_bouts()` respectively from the `endpoints/bradykinesia_endpoints.py` file. Regarding bradykinesia measures, our work showed that the hand movement amplitude feature had the strongest agreement with clinical measures and was able to discriminate between treatment related changes in motor states the best.

The figure below shows an example visualization of these digital measurements for a given day of data from a subject (a. Tremor related measures for subject, b. Bradykinesia related measures for subject).
<p align="center">
  <img width="800" height="700" src="https://raw.githubusercontent.com/NikhilMahadevan/analyze-tremor-bradykinesia-PD/master/images/continuous_digital_measures.png?token=ABFEV6WKEBNQ7Z4TYWACE5K5L7YOQ">
</p>

## License
This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/NikhilMahadevan/analyze-tremor-bradykinesia-PD/blob/update-readme/LICENSE) file for details