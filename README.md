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

The competition data is downloaded and explored locally, sheding light on the correlation between patient medical history, melanoma image and diagnosis outcome. This analysis is performed in the [explore/1.Data Preprocessing](https://github.com/mcachosoblechero/SpringBoard_Capstone/blob/main/1.%20Data%20Preprocessing.ipynb) jupyter notebook. The following aspects are addressed:
- Metadata data wrangling, analysing both missing and outlying values 
- Metadata EDA, exploring the correlation with the label as well as cross-correlation between metadata features
- Images EDA, exploring distinct features present in the provided images
- Images data augmentation techniques, that have the potential to improve model accuracy
These processing steps are performed locally.

The processing functions yielding from this analysis were refactored for production in [melanoma/data_preprocessing.py](https://github.com/mcachosoblechero)

## 3. Model Development

*Coming soon*
