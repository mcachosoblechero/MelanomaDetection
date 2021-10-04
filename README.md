# SpringBoard_Capstone

## 0. Initial Repositories Considered

As my area of expertise lies in Healthcare and Wearable devices, I would like my capstone project to align with this career vision. However, the main limitation for healthcare applications is the availability of publicly available clinical data.

Exploring Kaggle challenges, I found few datasets that might align with my vision for this project.
- Melanoma Detection: https://www.kaggle.com/c/siim-isic-melanoma-classification/overview
- Chest radiographs for COVID Detection: https://www.kaggle.com/c/siim-covid19-detection/overview
- Chest X-Ray Images for Pneumonia Detection: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

Based on the quality and the availability of data, as well as the potential of the dataset, I selected the <b>Melanoma Detection dataset</b>

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

The competition data is downloaded and explored, sheding light on the correlation between patient medical history, melanoma image and diagnosis outcome. This analysis is performed in the [1.Data Preprocessing](https://github.com/mcachosoblechero/SpringBoard_Capstone/blob/main/1.%20Data%20Preprocessing.ipynb) jupyter notebook. The following aspects are addressed:
- Metadata data wrangling, analysing both missing and outlying values 
- Metadata EDA, exploring the correlation with the label as well as cross-correlation between metadata features
- Images EDA, exploring distinct features present in the provided images
- Images data augmentation techniques, that have the potential to improve model accuracy

This processing step is performed locally.
