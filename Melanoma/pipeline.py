
import logging
import os
import sys
import pydicom
import data_preprocessing
import image_preprocessing
import pandas as pd


def run_pipeline():

    # Create basic logging configuration
    logging.basicConfig(
        format='[%(asctime)s|%(module)s.py|%(levelname)s]  %(message)s',
        datefmt='%H:%M:%S',
        level=logging.INFO,
        # filename='example.log'
        stream=sys.stdout
    )
    logging.info('Starting the data analysis pipeline')
    logging.info('Finished the data analysis pipeline')

    # Define datasets locations
    input_dcm_train = 'Datasets/dcm/train/'
    input_dcm_filelist = os.listdir(input_dcm_train)
    input_metadata_train = 'Datasets/train.csv'

    # Load metadata datasets and remove NA and duplicates
    md_train = pd.read_csv(input_metadata_train)                            \
        .pipe(lambda df: df.dropna(subset=["age_approx", "sex"], axis=0))   \
        .pipe(data_preprocessing.AgeOutlierAnalysis)                        \
        .pipe(data_preprocessing.RemoveDuplicateRecords)

    # # Test Image Resize and Augmentation
    # img = pydicom.dcmread(input_dcm_train + os.sep + input_dcm_filelist[0])
    # img_resized = image_preprocessing.ResizeImage(img.pixel_array, [256, 256])
    # filters = ["Gauss", "HFlip", "VFlip", "Micro", "Hair"]
    # for filter in filters:
    #     image_preprocessing.ImgAug(img_resized, filter, True)


if __name__ == '__main__':
    print(os.path.realpath(__file__))
    run_pipeline()
