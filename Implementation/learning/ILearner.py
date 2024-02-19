from typing import Any
import random
import math
import pandas as pd
import numpy as np
class ILearner:
    
    def __init__(self, options: Any) -> None :
        self.options = options
        self.learning_options = options["learning"]
        self.learning_params = self.learning_options["parameters"]["general"]
        self.learning_locations = self.learning_options["locations"]
        random.seed(3008)
        return
    
    
    def _generate_new_full_dataset(self) -> None :
        """Function that creates a new split in training and validation data"""
        
        # Check which data to used and create dataframe
        data_list = []
        if self.learning_params["training_uka"] == 1:
            location = self.options["execution"]["locations"]["uka"]
            data_list.append(pd.read_parquet(location, engine='auto'))
        if self.learning_params["training_eICU"] == 1:
            location = self.options["execution"]["locations"]["eICU"]
            data_list.append(pd.read_parquet(location, engine='auto'))
        if self.learning_params["training_MIMICIV"] == 1:
            location = self.options["execution"]["locations"]["MIMICIV"]
            data_list.append(pd.read_parquet(location, engine='auto'))
        data = pd.concat(data_list)

        # Fill Nans with value not possible because RF can not handle NaN
        data = data.fillna(-100000)
        data = data.reset_index()
        del data['index']

        # Get admissions where ARDS was present
        data_ards = data.query("ARDS == 1").reset_index()
        del data_ards["index"]

        # Get admissions where ARDS was not present
        data_no_ards = data.query("ARDS == 0").reset_index()
        del data_no_ards["index"]

        # Determine number of ARDS and non ARDS admissions in the data
        num_ards_real = len(data_ards.index)
        num_no_ards_real = len(data_no_ards.index)

        # Determine number of ARDS and non ARDS admission needed in dataset to meet the wished ratio
        total_calculated_ards = math.ceil(num_ards_real*(1/self.learning_params["ratio_ards_no_ards"]))
        num_data_no_ards_target = total_calculated_ards - num_ards_real
        total_calculated_no_ards = math.ceil(num_no_ards_real*(1/(1-self.learning_params["ratio_ards_no_ards"])))
        num_data_ards_target = total_calculated_no_ards-num_no_ards_real

        # Randomly sample data to achieve wished ratios
        if num_data_no_ards_target > num_no_ards_real:
            list_ards_indicies = random.sample(range(num_ards_real), num_data_ards_target)
            data_ards = data_ards.iloc[list_ards_indicies].reset_index()
            del data_ards['index']
        else :
            list_no_ards_indicies = random.sample(range(num_no_ards_real), num_data_no_ards_target)
            data_no_ards = data_no_ards.iloc[list_no_ards_indicies].reset_index()
            del data_no_ards["index"]

        self._generate_split_dataset(data_ards, data_no_ards)

        return

    def _generate_split_dataset(self, data_ards: pd.DataFrame, data_no_ards: pd.DataFrame) -> None:
        """Function that randomly splits dataset into training and validation"""

        # Get number of ARDS and non ARDS admission
        num_ards = len(data_ards.index)
        num_no_ards = len(data_no_ards.index)
        print("ARDS " + str(num_ards))
        print("NO ARDS: " + str(num_no_ards))
        print("TOTAL: " + str(num_ards+num_no_ards))

        # Calculate number of admissions in the validation set
        num_ards_test = math.floor(num_ards*self.learning_params["ratio_test_training"])
        num_no_ards_test = math.floor(num_no_ards*self.learning_params["ratio_test_training"])
        
        # Randomly sample data to create validation set
        list_indicies_ards_test = random.sample(range(num_ards), num_ards_test)
        list_indicies_no_ards_test = random.sample(range(num_no_ards), num_no_ards_test)
        data_ards_test = data_ards.iloc[list_indicies_ards_test].reset_index()
        del data_ards_test["index"]
        data_no_ards_test = data_no_ards.iloc[list_indicies_no_ards_test].reset_index()
        del data_no_ards_test["index"]
        test = pd.concat([data_no_ards_test, data_ards_test])

        # Put data not in validation set in training set
        data_ards_training = data_ards[~data_ards.index.isin(list_indicies_ards_test)]
        data_no_ards_training = data_no_ards[~data_no_ards.index.isin(list_indicies_no_ards_test)]
        training = pd.concat([data_no_ards_training, data_ards_training])
        
        # Save training and validation set
        print(len(training.index))
        print(len(test.index))
        test.to_parquet(self.learning_locations["test_set_location"], engine='auto', compression='snappy', index=None)
        self._write_data(test, self.learning_locations["test_set_location"])
        self._write_data(training, self.learning_locations["training_set_location"])
        self._write_data(training, "../Data/Training_Data/dump_feature_selection")
    
    
    
    def _write_data(self, data: pd.DataFrame, location) :
        """Function that is used to write data to .csv and parquet files"""

        # Build location strings
        location_pq = location + ".parquet"
        location_csv = location + ".csv"

        # Write data
        data.to_parquet(location_pq, engine='auto', compression='snappy', index=None)
        data.to_csv(location_csv, sep=",", index=None)

    # Functions for reading to from parquet files
    def _read_training_data(self) -> pd.DataFrame:
        return self._read_data(self.learning_locations["training_set_location"]+".parquet")

    def _read_data(self, location) -> pd.DataFrame:
        """Function that reads parquet data from location and split it into predictor and label"""
        data = pd.read_parquet(location, engine='auto')
        data = data.reset_index()
        del data['index']
        label = data["ARDS"]
        predictors = data.loc[:, data.columns != 'ARDS']
        return predictors, label
    
    def _read_test_data(self) -> pd.DataFrame:
        return self._read_data(self.learning_locations["test_set_location"]+".parquet")
    
    def _learn(self):
        pass

    def evaluate(self):
        pass
        
