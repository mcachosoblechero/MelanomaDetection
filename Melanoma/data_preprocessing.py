import pandas as pd


def RemoveDuplicateRecords(dataset):
    # """
    # Use information provided by competition host to remove duplicated records \
    # Input: Patient Metadata Dataframe
    # Output: Return clean Dataframe
    # """
    duplicates = pd.read_csv(
        'Datasets/resources/2020_Challenge_duplicates.csv')
    return dataset.drop(dataset.loc[dataset.image_name.isin(
        duplicates.ISIC_id_paired)].index)
