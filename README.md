# analyze-tremor-bradykinesia-PD
A Python (2.7) package that enables digital measurements of **resting tremor** and **bradykinesia** in patients with Parkinson's Disease. These measurements utilize accelerometer data as input from a single wrist-worn wearable device located on the most affected side.

## Overview
Objective assessment of Parkinson’s disease symptoms during free-living conditions can provide valuable information for disease management and help accelerate the development of new therapies. Traditional assessments are episodic (require patients to come into a clinic at specified intervals for assessment), low-fidelity (assessments are paper-based rudimentary numerical rating scales), and subjective in nature (assessments are reported by clinicians and patients). Recent advances in wearable sensor technology have enabled the use of objective digital measurements to assess Parkinson's disease symptoms in free-living conditions. Current digital measurements often require the use of multiple devices or performance of prescribed motor activites, which is not optimal for monitoring free-living conditions.

Herein we present our source code (Python 2.7) used for the development and validation of a method aimed at objective assessment of **resting tremor** and **bradykinesia** (two common symptoms of Parkinson's disease) using accelerometer data captured with a single wrist-worn device during the performance of unscripted activities. Our method combines context detection and symptom assessment by using heuristic and machine learning models in a hierarchical framework to provide continuous monitoring by sequentially processing epochs of raw sensor data. Results of our analysis show that sensor derived continuous measures of resting tremor and bradykinesia achieve good to strong agreement with clinical assessment of symptom severity and are able to discriminate between treatment related changes in Parkinsonian motor states (ON/OFF).

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
This repository contains code to create multiple classifiers (**_provided the user has data and ground truths_**) that fit together in a system as shown below:

![alt text](pd_analytics_diagram.png?raw=true "pd_analytics_diagram.png")

The repository is organized in the following fashion:
* __classifiers__: code to generate classifiers in each node of the tree above
* __endpoints__: code to aggregate endpoints for resting tremor and bradykinesia
* __features__: accelerometer signal features used to train supervised learning machine learning models
* __signal_preprocessing__: accelerometer signal preprocessing functions

