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
