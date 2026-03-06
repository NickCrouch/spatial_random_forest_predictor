# Spatial Random Forest Classification

This project demonstrates how to perform a spatial machine learning analysis using a Random Forest classifier in Python.
The workflow includes generating spatial data, training a model, evaluating performance, and interpreting feature importance.
Random forests are powerful machine learning models because they combine the predictions of many decision trees to improve accuracy and reduce overfitting.
By averaging across multiple trees built from different subsets of the data, random forests are able to capture complex nonlinear relationships and interactions between variables while remaining robust to noise and outliers.

## Overview

Random Forest models are widely used in spatial analysis tasks such as:

* Species distribution modeling
* Habitat suitability mapping
* Land cover classification
* Environmental risk prediction

This repository provides a reproducible example pipeline for applying a Random Forest classifier to spatial data.

## Workflow

The analysis follows these steps:

* Generate or load spatial data
* Define predictor variables and target labels
* Split the dataset into training and testing sets
* Train a Random Forest classifier
* Evaluate model performance
* Analyze feature importance
* Compute ROC AUC and plot the ROC curve

## Data

The example dataset contains synthetic spatial observations with the following features:

* latitude
* longitude
* elevation
* temperature
* precipitation

The target variable represents a binary environmental classification.

## Example output

Accuracy: 0.78 \
ROC AUC: 0.82

 | Feature | Importance |
 | ------------- | ------------- |
 | elevation | 0.26 |
 | precipitation | 0.25 |
 | temperature | 0.22 |
 | longitude | 0.14 |
 | latitude | 0.13 |

As the ROC curve below shows, in combination with the accuracy, this model does not have excellent performance.
This is likely due to the manner in which the data were generated for this analysis; however, the method is a simple and robust one for the analysis of spatial data.

<img width="600" height="500" alt="roc_curve" src="https://github.com/user-attachments/assets/57c2cfd3-fde5-45e1-aacf-5548d01a9852" />

