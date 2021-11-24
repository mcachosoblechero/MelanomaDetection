# Melanoma Detection

This repository contains a Melanoma Detection pipeline to identify malignant melanoma using both patient metadata and medical images.
To reproduce the results, download this repository and perform the following steps.

## 1. Data acquisition

The SIIM-ISIC Melanoma Classification Challenge in Kaggle can be downloaded using the Kaggle API, following these steps:
- Download Kaggle API to your Python Environment

<code>conda install -c conda-forge kaggle</code>
- Download the API key following the instructions found in https://www.kaggle.com/docs/api
- Accept the competition terms and conditions
- Download the full dataset using this API

<code>kaggle competitions download -c siim-isic-melanoma-classification</code>

The dataset, with a size of 108GB, is stored locally.

## 2. Data Wrangling and Exploratory Data Analysis

The competition data is downloaded and explored locally, sheding light on the correlation between patient medical history, melanoma image and diagnosis outcome. This analysis is performed in the [Explore/1.Data Preprocessing](https://github.com/mcachosoblechero/SpringBoard_Capstone/blob/main/1.%20Data%20Preprocessing.ipynb) jupyter notebook. The following aspects are addressed:
- Metadata data wrangling, analysing both missing and outlying values 
- Metadata EDA, exploring the correlation with the label as well as cross-correlation between metadata features
- Images EDA, exploring distinct features present in the provided images
- Images data augmentation techniques, that have the potential to improve model accuracy
These processing steps are performed locally.

The processing functions yielding from this analysis were refactored for production in [melanoma/data_preprocessing.py](https://github.com/mcachosoblechero)

## 3. Model Development

Based on the understanding developed throughout the Exploratory Data Analysis, I test a set of Machine Learning and Deep Learning models with both metadata and image data. This analysis is performed in the [Explore/2. Model Development](https://github.com/mcachosoblechero/MelanomaDetection/blob/main/Explore/2.%20Model%20Development.ipynb) jupyter notebook. In this analysis, I evaluate both AUC and Recall of the following models:
- <b>Using Metadata</b>
    - Traditional ML models with standard configuration - KNN, Random Forest and XGBoost
    - Fine tune XGBoost using GridSearchCV
    - Simple Neural Network 
- <b>Using Melanoma Images</b>
    - Simple CNN with MaxPooling
    - Modified Learning Rate using Cosine Annealing with Hard Restarts
    - Simple CNN with image augmentation
    - Transfer Learning CNN, using EfficientNetB2

All these models are stored and re-evaluated in the [Explore/3. Model Selection](https://github.com/mcachosoblechero/MelanomaDetection/blob/main/Explore/3.%20Model%20Selection.ipynb) jupyter notebook. This analysis yields the selected model for deployment.

## 4. Model Deployment

The selected model is deployed as part of a user-friendly Desktop GUI. This GUI loads both the target image and the selected model, and provides a diagnosis. Below I include two examples of how this GUI operates.

![Benign Diagnosis](XX)
![Malignant Diagnosis](XX)

