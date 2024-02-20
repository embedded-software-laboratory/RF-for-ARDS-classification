import random
import math
import pandas as pd

"""This script is used to split up all extracted data from the different datasets and filters into training and test data set.
 The ratio between  test and training set is 1:4 it is ensured that each dataset consists of close to 10% ARDS patients 
 and 90% non-ARDS patients"""


def _generate_new_full_dataset(data_to_split) -> tuple[pd.DataFrame, pd.DataFrame]:
    """This function is used to calculate the number of patients of each kind  that needs to be present in the training
    and test set. Furthermore, it calls the function that generates both data sets.

    Params:
        data_to_split (pd.DataFrame): The data that needs to be split into training and test set

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:  Containg at the first position the training set, second position contains
                                            the test set"""


    # Fill NaNs with value that does not occur in our data because our RF can not work with NaNs and ensure coherent index
    data_to_split = data_to_split.fillna(-100000)
    data_to_split = data_to_split.reset_index()
    del data_to_split['index']

    # Split dataset in dataset containing ARDS patients and non-ards patients
    data_ards = data_to_split.query("ARDS == 1").reset_index()
    del data_ards["index"]
    data_no_ards = data_to_split.query("ARDS == 0").reset_index()
    del data_no_ards["index"]

    # Get number of patients of both groups and calculate how many patients are needed to ensure a ratio of 1:10 between
    # ARDS and non ARDS patients.
    num_ards_real = len(data_ards.index)
    num_no_ards_real = len(data_no_ards.index)
    total_calculated_ards = math.ceil(num_ards_real * (1 / 0.1))
    num_data_no_ards_target = total_calculated_ards - num_ards_real
    total_calculated_no_ards = math.ceil(num_no_ards_real * (1 / (1 - 0.1)))
    num_data_ards_target = total_calculated_no_ards - num_no_ards_real

    # Ensure ratio can be achieved by randomly downsampling overrepresented group
    if num_data_no_ards_target > num_no_ards_real:
        list_ards_indicies = random.sample(range(num_ards_real), num_data_ards_target)
        data_ards = data_ards.iloc[list_ards_indicies].reset_index()
        del data_ards['index']
    else:
        list_no_ards_indicies = random.sample(range(num_no_ards_real), num_data_no_ards_target)
        data_no_ards = data_no_ards.iloc[list_no_ards_indicies].reset_index()
        del data_no_ards["index"]

    # Split dataset
    return _generate_split_dataset(data_ards, data_no_ards)


def _generate_split_dataset(data_ards: pd.DataFrame, data_no_ards: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """This function is used to split the patients into training and test set

    Params:
        data_ards (pd.DataFrame): DataFrame that contains the data of all patients diagnosed with ARDS
        data_no_ards (pd.DataFrame): DataFrame that contain the data of all patients not diagnosed with ARDS

    Returns:
        tuple: Tuple containing at the first position the training and at the second position the test set
    """

    # Print information on dataset
    num_ards = len(data_ards.index)
    num_no_ards = len(data_no_ards.index)
    print("ARDS " + str(num_ards))
    print("NO ARDS: " + str(num_no_ards))
    print("TOTAL: " + str(num_ards + num_no_ards))

    # Calculate number of patients of each group needed in the test set
    num_ards_test = math.floor(num_ards * 0.25)
    num_no_ards_test = math.floor(num_no_ards * 0.25)

    # Randomly sample from data to create test set, ensure coherent indicies
    list_indicies_ards_test = random.sample(range(num_ards), num_ards_test)
    list_indicies_no_ards_test = random.sample(range(num_no_ards), num_no_ards_test)
    data_ards_test = data_ards.iloc[list_indicies_ards_test].reset_index()
    del data_ards_test["index"]
    data_no_ards_test = data_no_ards.iloc[list_indicies_no_ards_test].reset_index()
    del data_no_ards_test["index"]
    test_set = pd.concat([data_no_ards_test, data_ards_test])
    data_ards_training = data_ards[~data_ards.index.isin(list_indicies_ards_test)]
    data_no_ards_training = data_no_ards[~data_no_ards.index.isin(list_indicies_no_ards_test)]

    # Create training set from leftover patients and ensure coherent indicies
    training_set = pd.concat([data_no_ards_training, data_ards_training])
    print(len(training_set.index))
    print(len(test_set.index))
    return training_set, test_set


def _write_data(data: pd.DataFrame, location):
    """Function that writes Dataframe to specific location as .parquet and .csv file
        Params:
            data_to_write (pd.DataFrame): Dataframe that should be saved to file
            location_to_write (str): Location where file is saved
    """

    location_pq = location + ".parquet"
    location_csv = location + ".csv"
    data.to_parquet(location_pq, engine='auto', compression='snappy', index=None)
    data.to_csv(location_csv, sep=",", index=None)


def _delete_admissions(data: pd.DataFrame, to_delete: list) -> pd.DataFrame:
    """This function deletes entries from the dataframe if their index is in the provided list.
        Params:
            data_to_clean (pd.DataFrame): Dataframe containing the patient data
            to_delete (list): List containing the indicies that need to be deleted
        Returns:
            pd.DataFrame: Dataframe that contains the patient data without deleted entries"""

    to_delete_set = set(to_delete)
    data = data.drop(to_delete_set)
    data.reset_index(inplace=True)
    del data['index']
    return data


def filter_admissions_learning(data_to_filter: pd.DataFrame, high, low, contra) -> pd.DataFrame:
    """Function that is used to filter the data before it is split into training and test sets.
        Params:
            data (pd.DataFrame): Contains the data to be filtered
            high (int): Indicates whether patients that are diagnosed with ARDS but do not have a horovitz lower than 300 should
            be filtered
            low (int): Indicates whether patients that are not diagnosed with ARDS but do have a horovitz lower than 200 should be
            filtered
            contra (int): Indicates whether patients that not are diagnosed with ARDS and do not have a diagnosis that could explain
            a horovitz below 200 should be filtered

        Returns:
            pd.Dataframe: contains the filtered data"""

    # Initialization of informative variables
    to_delete = []
    counter_High_ARDS = 0
    counter_Low_No_ARDS = 0
    counter_Low_No_Contra = 0

    # Filter data
    for index, row in data_to_filter.iterrows():
        if high == 1:
            # Filter patients with ARDS diagnosis but a too high horovitz
            if row['ARDS'] == 1 and row['Minimaler_Horowitz'] > 300:
                to_delete.append(index)
                counter_High_ARDS = counter_High_ARDS + 1

        if low == 1:
            # Filter patients with no ARDS diagnosis but a too low horovitz
            if row["ARDS"] == 0 and row["Minimaler_Horowitz"] < 200:
                to_delete.append(index)
                counter_Low_No_ARDS = counter_Low_No_ARDS + 1

        elif contra == 1:
            # Filter patients with no ARDS or other diagnosis that could explain the low horovitz
            contraindidaction = 1 if (
                        row["Lungenoedem"] == 1 or row["Herzinsuffizienz"] == 1 or row["Hypervolaemie"] == 1) else 0
            if contraindidaction == 0 and row["ARDS"] == 0 and row["Minimaler_Horowitz"] < 200:
                to_delete.append(index)
                counter_Low_No_Contra = counter_Low_No_Contra + 1

    print("HIGH ARDS:" + str(counter_High_ARDS))
    print("LOW NO ARDS: " + str(counter_Low_No_ARDS))
    print("LOW NO ARDS NO Contra: " + str(counter_Low_No_Contra))
    print("Before :" + str(len(data_to_filter.index)))
    data_filtered = _delete_admissions(data_to_filter, to_delete)
    print("After :" + str(len(data_filtered.index)))

    return data_filtered


if __name__ == '__main__':
    """This function starts the generation of the training and test data sets"""

    # Define datasets and filters for which training/test sets should be generated
    databases = ["uka", "eICU", "MIMICIV"]
    filters = ["full", "extreme", "light"]
    for db in databases:

        for f in filters:
            # Read Extracted Data
            file = "../Data/Extracted_Data/" + db + "_data.parquet"
            data = pd.read_parquet(file, engine='auto')

            # Create paths to location where sets will be saved
            location = db + "_data_" + f
            print(location)
            location_training = "../Data/Training_Data/" + location
            test_location = "../Data/Test_Data/" + location

            # Filter patients according to chosen filter
            if f == "extreme":
                data = filter_admissions_learning(data, 1, 1, 0)

            elif f == "light":
                data = filter_admissions_learning(data, 1, 0, 1)
            else:
                data = data

            # Split data into sets and save generated sets
            training, test = _generate_new_full_dataset(data)
            _write_data(test, test_location)
            _write_data(training, location_training)
