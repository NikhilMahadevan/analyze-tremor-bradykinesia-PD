# analyze-tremor-bradykinesia-PD
This repository contains Python code that can be used to build analytics to measure aspects of tremor and bradykinesia in patients with Parkinson's Disease. These analytics utilize accelerometer data from a wrist-worn wearable device.

## Overview
Objective assessment of Parkinson’s disease symptoms during daily life can provide valuable information for disease management and help accelerate the development of new therapies. Traditional approaches often require the use of multiple devices or performance of prescribed motor activities, which is not optimal for monitoring free-living conditions. Recent advances in wearable sensor technology have enabled objective assessment of Parkinson's disease symptoms in free-living conditions. This package contains source code (Python) to assist in development of **resting tremor** and **bradykinesia** analytics based on accelerometer data captured from a wrist-worn wearable device.

## Repository Contents
This repository contains code to create multiple classifiers (provided the user has data and ground truths) that fit together in a system as shown below:

![alt text](pd_analytics_diagram.png?raw=true "pd_analytics_diagram.png")

The repository is organized in the following fashion:
* __classifiers__: code to generate classifiers in each node of the tree above
* __endpoints__: code to aggregate endpoints for resting tremor and bradykinesia
* __features__: accelerometer signal features used to train supervised learning machine learning models
* __signal_preprocessing__: accelerometer signal preprocessing functions