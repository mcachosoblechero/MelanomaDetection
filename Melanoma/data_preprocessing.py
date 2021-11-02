import pandas as pd


def AgeOutlierAnalysis(dataset):

    # """
    # Look for outliers on the age ranges, and correct it if possible
    # Input: Patient Metadata Dataframe
    # Output: Return outlier-free Dataframe
    # """

    # Valid date assumption
    Age_ValidRange = [5, 95]

    # Discover outliers
    Age_OutliersMask = (dataset.age_approx <= Age_ValidRange[0]) | \
        (dataset.age_approx >= Age_ValidRange[1])
    Age_Outliers = dataset.loc[Age_OutliersMask]

    # For each outlier:
    #   Check patient record
    #   If available, replace by mean
    #   Otherwise, drop
    for id in Age_Outliers.patient_id.unique():
        PatientMask = (dataset.patient_id == id) & ~(Age_OutliersMask)
        PatientRecords = dataset.loc[PatientMask]

        if len(PatientRecords) > 0:
            dataset.loc[Age_OutliersMask, "age_approx"] \
                = PatientRecords.age_approx.mean()
        else:
            dataset = dataset.loc[~(dataset.patient_id == id)]

    return dataset


def RemoveDuplicateRecords(dataset):
    # """
    # Use information provided by competition host to remove duplicated records \
    # Input: Patient Metadata Dataframe
    # Output: Return clean Dataframe
    # """
    duplicates = pd.read_csv(
        'Datasets/resources/2020_Challenge_duplicates.csv')
    return dataset.drop(dataset.loc[dataset.image_name.isin(duplicates.ISIC_id_paired)].index)
