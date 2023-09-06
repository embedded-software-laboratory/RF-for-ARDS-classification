from typing import Any
import pandas as pd
import numpy as np
from numpy import ndarray
from Lookup.MIMICIV_item_dict import *
from Lookup.uka_extraction_indicies import *
from Lookup.uka_data_indicies import *
from Lookup.features_new import *
from Lookup.static_information import *
from preprocessing.IPreprocessor import IPreprocessor



"""Class that is used to preprpcess Data extracted from the Extractor. The class provides various functions already used during extraction.
The most important function is process_data which takes the extracted data and returns the feature vector"""
class ukaPreprocessor(IPreprocessor):
    def __init__(self, options: Any, job_id: int):
        # Init superclass and class variables
        super().__init__(options, job_id)
        self.options = options

    def process_data(self, data: pd.DataFrame, admission: ndarray,  min_horowitz_time:float, max_time: float, min_time:float) :
        """Function that creates a dataframe which has values on all parameters at all times. If no information is know for a timestamp its value is NaN.
         Information on admission, as well times important for the windows is needed"""
        
        # Process information that is presumed to be mostly unchanged during an admission
        static_info = self._assign_static_information(admission, min_horowitz_time, uka_code_reason)
        
        # Create dataframe containing all timepoints in the window in 15 minutes intervalls
        data.rename(columns={'Zeit_ab_Aufnahme' : 'Zeitpunkt'}, inplace=True)
        time_table = pd.DataFrame(np.arange(min_time, max_time+15, 15), columns=['Zeitpunkt'])
        
        # Create list of DFs where each DF contains a full time table and stores information for 
        # one parameter in the uka Database. Each list entry corresponds to one parameter
        data_list = [pd.DataFrame(columns=['Zeitpunkt', key]) for key in uka_name_index_dict]
        
        # Create DF that contains all parameters and has a full time table
        variables = self.generate_variable_page()
        variables['Zeitpunkt'] = list(time_table['Zeitpunkt'])

        # Initalize each entry in the list by assigning the relevant data from the extracted data
        for column in data:
            if not (column =='patientid' or  column == 'Zeitpunkt'):
                # Get data for parameter from extracted data
                relevant_data = data[['Zeitpunkt', column]]
                
                # Drop entries where there is no information
                relevant_data = relevant_data.dropna(axis=0)

                # Store data in right list entry
                data_list[uka_name_index_dict.get(column)] = relevant_data
                
        # Impute information regarding ventilation parameters
        for index in range(len(uka_ventilation)):
            
            # Get ventilation data from list
            name = uka_ventilation[index]
            data_vent = data_list[uka_name_index_dict.get(name)]
            
            # Impute missing ventilation data and store it replace old ventilation parameter with imputated data
            processed_data = self._impute_find_indexes(data_vent, name, 'Zeitpunkt', uka_ventilation_cutoff.get(name), 15, np.nan)
            data_list[uka_name_index_dict.get(name)] = processed_data
        
        # Impute data for IV drugs
        for index in range(len(uka_drugs)):

            # Get drug data from list
            data_drug = data_list[uka_name_index_dict.get(uka_drugs[index])]

             # Impute missing drug data and store it replace old drug parameter with imputated data
            processed_data = self._impute_find_indexes(data_drug, uka_drugs[index], 'Zeitpunkt', 60, 15, np.nan)
            data_list[uka_name_index_dict.get(uka_drugs[index])] = processed_data
        
        # Calculate the difference between P EI and PEEP
        data_list[uka_name_index_dict.get('deltaP')] = self._calculate_deltaP(data_list[uka_name_index_dict.get('PEEP')], data_list[uka_name_index_dict.get('P_EI')])

        for index in range(len(data_list)):
            merged = time_table.merge(data_list[index], how='left', on='Zeitpunkt')
            variables[uka_index_name_dict.get(index)] = merged[uka_index_name_dict.get(index)]
            
        # Reset defragment Dataframe for performance improvement
        new_variables = variables.copy()
        variables = new_variables

        
        
        # Convert data to feature vector 
        feature, freq = self._assign_features(variables, static_info, feature_position_dict_uka)
        
        return feature, freq


    


   
    
    def _assing_minimal_Horowitz(self, data: pd.DataFrame) -> float:
        return data.min(axis=1).at[0]
    
    def calculateHorowitz(self, data: pd.DataFrame) -> pd.DataFrame:
        """Function that returns the time where the lowest horowitz value was observerd. Takes a dataframe containing horovitz-indexes as input"""

        # Ensure proper nameming
        data.rename(columns={'time' : 'abstime'}, inplace=True)
        
        # Find index of lowest horovitz-index
        idx_min_avg_horowitz, min_horowitz = self.find_lowest_Horowitz(data)
        
        # Ensure proper nameming
        data.rename(columns={'abstime' : 'time'}, inplace=True)
        
        # Find time of minimal horovitz-index
        min_horowitz_time = data['time'][idx_min_avg_horowitz]
        return min_horowitz_time, min_horowitz
    
    def _calculate_deltaP(self, data_PEEP: pd.DataFrame, data_P_EI: pd.DataFrame) -> pd.DataFrame:
        """Function that calculate the difference between P EI and PEEP"""

        # Merge data for P EI and PEEP into one dataframe and sort it according to their timestamp
        data = pd.merge(data_P_EI, data_PEEP, how='outer', on='Zeitpunkt')
        data = data.sort_values(by='Zeitpunkt')
        data = data.reset_index()
        del data['index']

        # Calculate delta P and store timestamps and values in a dataframe
        deltap_times, deltap_values = super()._calculate_deltaP(data, 'PEEP', 'P_EI', 'Zeitpunkt')
        deltap = pd.DataFrame(columns=['Zeitpunkt', 'deltap'])
        deltap['Zeitpunkt'] = deltap_times
        deltap['deltaP'] = deltap_values


        return deltap

     

    
