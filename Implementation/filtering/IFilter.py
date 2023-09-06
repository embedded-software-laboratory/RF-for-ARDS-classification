from typing import Any


from numpy import ndarray
import numpy as np
import pandas as pd



class IFilter:
    def __init__(self, options: Any):
        self.options = options
    
    
    # Takes a ndarray, the position of the identifier in said array and a list of ids that should not be deleted
    def filter_ids_extract_admissions(array: ndarray, id_position: int, leftover: list) -> ndarray:
        """Function that is used to discard admissions from an array"""
        
        # Create dictonary of admission id to index
        id_dict = {}
        for index, entry in enumerate(array):

            id_dict[entry[id_position]] = index
        
        # Get all ids that need to be deleted
        to_delete_ids = set(id_dict.keys()) - set(leftover)

        # Get array index of ids that need to be deleted
        to_delete_index = [id_dict.get(entry) for entry in to_delete_ids]

        # Delete admissions
        array = np.delete(array, to_delete_index, axis=0)
        return array

    def filter_ids_extract_data(array: ndarray, id_position: int, leftover: list) -> ndarray:
        """Function that is used to delete admissions with no horovitz-index from array"""

        # Create dictonary of admission id to index
        id_dict = {}
        for index, entry in enumerate(array):

            id_dict[int(str(entry[id_position])[1:-1])] = index
        
        # Get all ids that need to be deleted 
        to_delete_ids = set(id_dict.keys()) - set(leftover)

        #  Get array index of ids that need to be deleted
        to_delete_index = [id_dict.get(entry) for entry in to_delete_ids]

        # Delete admissions
        array = np.delete(array, to_delete_index, axis=0)
        return array

    def filter_admissions_learning(self, data:pd.DataFrame):
        """Function that applies different filters before the data is used for training"""
        
        # Get filter settings
        options_filtering = self.options["filtering"]
        print(options_filtering)
        options_active_filters = options_filtering["parameters"]
        
        # Init variables
        to_delete = []
        counter_High_ARDS = 0
        counter_Low_No_ARDS = 0
        counter_Low_No_Contra = 0

        # Check every row if it needs to be filtered out
        for index, row in data.iterrows():

            # Filter out admissions where ARDS was diagnosed but lowest Horovitz-index is greater than 300mmHg and thus not meeting diagnose requirements
            if options_active_filters["filter_ARDS_high_Horowitz"] == 1:
                if row['ARDS'] == 1 and row['Minimaler_Horowitz'] > 300:
                    to_delete.append(index)
                    counter_High_ARDS = counter_High_ARDS + 1

            # Filter out admissions where ARDS was not diagnosed but lowest Horovitz-index is lower than 200mmHg and thus ARDS is possible
            if options_active_filters["filter_no_ARDS_low_Horowitz"] == 1:
                if row["ARDS"] == 0 and row["Minimaler_Horowitz"] < 200:
                    to_delete.append(index)
                    counter_Low_No_ARDS = counter_Low_No_ARDS +1

            # Filter out admissions where ARDS was not diagnosed but lowest Horovitz-index is lower than 200mmHg and no contraindications are present, thus making  a ARDS diagnosis reasonable
            elif options_active_filters["filter_no_ARDS_low_Horowitz_contraindication_present"] == 1:
                contraindidaction = 1 if (row["Lungenoedem"] == 1 or row["Herzinsuffizienz"] == 1 or row["Hypervolaemie"] == 1) else 0
                if contraindidaction == 0 and row["ARDS"] == 0 and row["Minimaler_Horowitz"] < 200:
                    to_delete.append(index)
                    counter_Low_No_Contra = counter_Low_No_Contra +1
                
                    
        # Print information on filtered numbers
        print("HIGH ARDS:" + str(counter_High_ARDS))
        print("LOW NO ARDS: " + str(counter_Low_No_ARDS))
        print("LOW NO ARDS NO Contra: " + str(counter_Low_No_Contra))
        print("Before :" + str(len(data.index)))
        data_filtered = self._delete_admissions(data, to_delete)
        print("After :" + str(len(data_filtered.index)))                
            
            


        return data_filtered

   
    def _delete_admissions(self, data: pd.DataFrame, to_delete: list) -> pd.DataFrame:
        """Function that is used to filter out admission and discard them"""
        
        to_delete_set = set(to_delete)
        data = data.drop(to_delete_set)
        data.reset_index(inplace= True)
        del data['index']
        return data
    
    def filter_admissions_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Not used in thesis"""
        
        options_filtering = self.options["execution"]["filtering"]
        options_active_filters = options_filtering["parameters"]
        to_delete = []
        for index, row in data.iterrows():
            if options_active_filters["filter_features_present"] == 1:
                    if row.isnull().sum(axis=1) < len(row.columns) * options_active_filters["filter_percentage_of_features"] :
                        to_delete.append(index)
        data_filtered = self._delete_admissions(data, to_delete)
        return data_filtered
