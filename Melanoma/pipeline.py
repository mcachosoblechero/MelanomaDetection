
import logging
import os
import sys
import data_preprocessing
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
    input_jpeg_train = 'Datasets/jpeg/train/'
    input_metadata_train = 'Datasets/train.csv'

    # Load metadata datasets and remove NA and duplicates
    md_train = pd.read_csv(input_metadata_train)                            \
        .pipe(lambda df: df.dropna(subset=["age_approx", "sex"], axis=0))   \
        .pipe(data_preprocessing.AgeOutlierAnalysis)

    md_train = data_preprocessing.RemoveDuplicateRecords(md_train)


if __name__ == '__main__':
    print(os.path.realpath(__file__))
    run_pipeline()
