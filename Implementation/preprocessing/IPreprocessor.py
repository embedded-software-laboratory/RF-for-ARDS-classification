import string
from typing import Any
import math


import numpy as np
from numpy import ndarray
import pandas as pd

from Lookup.features_new import *
from Lookup.all_features import *
from Lookup.uka_extraction_indicies import *
from Lookup.uka_data_indicies import *
from Lookup.static_information import *


class IPreprocessor:
    def __init__(self, options: Any, job_id: int) -> None:
        self.options = options
        self.job_id = job_id
        return
    
    
    

    # Creates a Dataframe containing all variable names as collumnnames
    @staticmethod
    def generate_variable_page() -> pd.DataFrame:
        return pd.DataFrame(columns=list_all_features)
    
    # Creates a Dataframe containing all feature names as collumnnames
    @staticmethod
    def generate_feature_page() -> pd.DataFrame:
        return pd.DataFrame(columns=list_features_var)
    
 
    

    def _assign_features(self, variable: pd.DataFrame, static_information: list, dict_windowsize: dict) -> tuple:
        """This function converts all processed data to a feature vector and counts the frequency of each parameter. 
        It needs the processed data and information on static data. Furthermore a dictonary containing the windosizes for each parameter is needed."""

        # Get DF containing all features
        feature = self.generate_feature_page()
        
        # Iterate over every parameter
        for column in variable:
            
            # Skip column which contains the timestamp
            if column == 'Zeitpunkt':
                continue
            
            # Get relevant data 
            data = variable[column]
            
            # Determine window size for data
            windowsize = dict_windowsize.get(column)
            
            
                
            list_data = []
            
            # Add staticdata
            if column in static_information_names:
                list_data = [static_information[static_dict.get(column)]]
                
            
            # Calculate mean of features that are only present once in the final feature vector
            elif windowsize == 0:
                list_data = [data.mean()]
            
            # Calculate feature values for features that are present more than once in the final feature vector
            # by taking the mean value for each window
            else:
                grouped = data.groupby(np.arange(len(data))//windowsize).mean()
                list_data = grouped.tolist()
                
                
            
            # Assign values to features in final vector
            if len(list_data) == 1 :
                feature[column] = [list_data[0]]
            else :
                for i in range(len(list_data)):
                    column_name = column + str(i)
                    feature[column_name] = [list_data[i]]
            
        
        
        # Replace non numerical values with numeric ones because the learning algorithm can not handle non numerical values
        feature = feature.replace([True, False], [1, 0])
        feature = feature.replace({'Geschlecht' : {"M" : 0, "W": 2, "F": 2, "O": 1, "U": 3, "": 3, "Unknown": 3, "Male": 0, "Female": 2, "Other": 1}})
        
        # Get all collumns where horowitz is stored
        relevant_columns = []
        for item in list_features_var:
            if "Horowitz-Quotient_(ohne_Temp-Korrektur)" in item:
                relevant_columns.append(item)
        
        # Find point with lowest horowitz within all features that store horowitz
        feature.loc[0, 'Minimaler_Horowitz'] = self._assing_minimal_Horowitz(feature[relevant_columns])
        
        # Count number of non-nan entries in the variable dataframe for every column
        freq = pd.DataFrame(variable.count()).transpose()

        return feature, freq

    def _assing_minimal_Horowitz(self, data: pd.DataFrame) -> float:
        return data.min(axis=1).at[0]

    def _convert_position(self, data, length) -> float :
        """Calculate the percentage of a patient being in the prone position in a timeframe"""
        value = data.sum(axis=0)/length
        return value

    def _assign_static_information(self, admission: ndarray, min_horowitz_timestamp: float, code_reason_dict: dict) -> list:
        """ Function that puts all information which was deemed to be static in a list and return that list. Needs information and admission
         minimum horowitz-index timestamp and a dictionary for the mapping of diagnosis codes to illnesses"""
        
        # Init list that contains static information
        static_info = []
        
        # Deal with missing heights or weights
        height = admission[HEIGHT] if not math.isnan(admission[HEIGHT]) else np.nan
        weight = admission[WEIGHT] if not math.isnan(admission[WEIGHT]) else np.nan
        
        # Calculate BMI if possible
        if not (math.isnan(height) or math.isnan(weight) or height==0):
            bmi = weight/(height/100*height/100)
        else :
            bmi = np.nan 
        
        # Add patient info 
        static_info.append(admission[AGE][0])
        static_info.append(admission[GENDER][0])
        static_info.append(weight)
        static_info.append(height)
        static_info.append(bmi)
        static_info.append(0)
        static_info.append(min_horowitz_timestamp)
        static_info.append(admission[ADMISSION_ID][0])
        
        
        # Assign relevant diagnosis according to the catolugoue of items
        diagnosis_info = self._assign_Diagnosis(admission[DIAGNOSIS], code_reason_dict)
        
        # Concat lists before returning
        static_info = static_info + diagnosis_info
        return static_info


    def _assign_Diagnosis(self, diagnosis: str, reason_code_dict:dict) -> list:
        """Function that maps the codes contained in the databases to their dieases. Needs dictionary 
        for correct mapping"""
        
        # Make list for all diagnosis set to false
        diagnosis_list = [False for _ in range(len(reason_index_dict))]

        # Check if a relevant diagnosis was recorded during an admission and set the according entry to True
        for key in reason_code_dict:
            if key in diagnosis:
                reason = reason_code_dict.get(key)
                diagnosis_list[reason_index_dict.get(reason)] = True
        return  diagnosis_list

   

   
    def find_lowest_Horowitz(self, patient_horowitz: pd.DataFrame):
        """Function that finds the lowest horovitz-value using a sliding window approach"""
        
        # Determines cutoff time for new horowitz series
        cutoff_horowitz_time = self.options["extraction_parameter"]["cutoff_time_horowitz"]
        
        # Calculate difference between two entries 
        patient_horowitz['time_diff'] = patient_horowitz['abstime'].rolling(2).apply(lambda x: x.iloc[1] - x.iloc[0])

        # Splits Dataframe if the Time difference is greater than the cutoff value
        split_patient_horowitz = []
        split_pos = 0
        for i in range(patient_horowitz.shape[0]):
            if not math.isnan(patient_horowitz['time_diff'][i]) and patient_horowitz['time_diff'][i]>cutoff_horowitz_time: 
                split_patient_horowitz.append(patient_horowitz.iloc[split_pos:i, :])
                split_pos = i
        split_patient_horowitz.append(patient_horowitz.iloc[split_pos:, :])
        
        # Calculates avg horowitz in a given window
        avg_horowitz = []
        for df in split_patient_horowitz:
            max_horowitz_entries = 0

            # Determines how many horowitzvalues  are averaged
            windowsize_horowitz = self.options["extraction_parameter"]["windowsize_horowitz"]
            length = len(df)

            # Make sure every split has a mean horowitz
            if max_horowitz_entries< length:
                max_horowitz_entries = length
            if max_horowitz_entries<windowsize_horowitz:
                windowsize_horowitz = max_horowitz_entries
            
            # Add average value for window to DF
            avg_horowitz.extend(df['Horowitz-Quotient_(ohne_Temp-Korrektur)'].rolling(windowsize_horowitz).mean())
        patient_horowitz['mean_horowitz'] = avg_horowitz
        
        # Get time of minimal horowitz
        
        idx_min_avg_horowitz = patient_horowitz['mean_horowitz'].idxmin()
        min_horowitz = patient_horowitz['mean_horowitz'].at[idx_min_avg_horowitz]
        return idx_min_avg_horowitz, min_horowitz

    def _process_Ventilation_features(self, data: pd.DataFrame, ventilation: pd.DataFrame) -> pd.DataFrame:
        """This function is used to impute Ventilation parameters. They are only stored when changes occur therefore all values in between changes have to be imputed"""
        
        # Initialize variables
        list_dfs = []
        list_dfs_processed = []
        
        # For each ventilation period add a df containing all timestamps within the ventilation timeframe
        for _, vent_row in ventilation.iterrows():
            data_times = pd.DataFrame(columns=['charttime'])
            data_times['charttime'] = [charttime for charttime in np.arange(vent_row['start'], vent_row['end'], 1.0)]
            list_dfs.append(data_times)

        
        # Each ventilation period merge the data available with the full timestamps of the ventilation period
        for df in list_dfs:
            
            # Merge data with timestamps
            df = df.merge(data[['charttime', 'value']], on='charttime', how='left')
            last_value = np.nan
            values = []

            # Fill empty data with last known value
            for index, row in df.iterrows():
                if not math.isnan(row['value']):
                    last_value = row['value']
                    
                values.append(last_value)
            df['value'] = values
            list_dfs_processed.append(df)
        return pd.concat(list_dfs_processed)

    def _calculate_deltaP(self, data: pd.DataFrame, PEEP_column: str, P_EI_column: str, time_column):
        """Calculates the difference between P EI and PEEP. Needs PEEP and PEI data in on DF and information on
        PEEP/PEI/timestamp column name """



        # Init variables containing information on delta P timestamp and value
        deltap_values = []
        deltap_times = []
        
        # Calculate delta P by searching for non nan PEI values and than finding the nearest PEEP value recorded before or with the PEI value
        # Than calulate delta P by subracting PEEP from PEI and storing the value and timestamp
        # Iterate over every entry in the data to find non-nan values for PEEP and PEI
        for index, row in data.iterrows():
            # Search non nan PEI values
            if not math.isnan(row[P_EI_column]):
                # Search next non PEEP value recorded before or with the current PEI value
                for i in range(index, 0, -1):
                    if not math.isnan(data[PEEP_column].at[i]):
                        # Calculate difference
                        deltap_times.append(row[time_column])
                        deltap_values.append(row[P_EI_column]-data[PEEP_column].at[i])
                    break
        return deltap_times, deltap_values             
    
    
    """Below are conversion functions for different parameters due to their simplicity they are not commented"""
    def _Hemoglobin(self, data: pd.DataFrame) -> pd.DataFrame:
        for i in range(len(data.index)):
            try: 
                data['value'].at[i] = data['value'].at[i] * 0.6206
            except:
                data['value'].at[i] = np.nan
        return data
    
    def _ident(self, values: list) -> pd.DataFrame:
        return values
    
    def _urea_nitrogen(self, data: pd.DataFrame) -> pd.DataFrame:
        for i in range(len(data.index)):
            try: 
                data['value'].at[i] = data['value'].at[i] * 0.1665
            except:
                data['value'].at[i] = np.nan
        return data

    def _creatinine(self, data: pd.DataFrame) -> pd.DataFrame:
        for i in range(len(data.index)):
            try: 
                data['value'].at[i] = data['value'].at[i] * 88.42
            except:
                data['value'].at[i] = np.nan
        return data
    
    def _albumin(self, data: pd.DataFrame) -> pd.DataFrame:
        for i in range(len(data.index)):
            try: 
                data['value'].at[i] = data['value'].at[i] * 151.5152
            except:
                data['value'].at[i] = np.nan
        return data

    def _fahrenheit(self, data: pd.DataFrame) -> pd.DataFrame:
        for i in range(len(data.index)):
            try:
               data['value'].at[i] = (data['value'].at[i]-32)*5/9
            except:
                data['value'].at[i] = np.nan
        return data

    def _etCO2(self, data: pd.DataFrame) -> pd.DataFrame:
        for i in range(len(data.index)):
            try: 
                data['value'].at[i] = data['value'].at[i] * 1/7
            except:
                data['value'].at[i] = np.nan
        return data

    def _CRP(self, data: pd.DataFrame, factor: float) -> pd.DataFrame:
        for i in range(len(data.index)):
            try: 
                data['value'].at[i] = data['value'].at[i] * factor
            except:
                data['value'].at[i] = np.nan
        return data

    def _Bilirubin_total(self, data: pd.DataFrame) -> pd.DataFrame:
        for i in range(len(data.index)):
            try:
                data['value'].at[i] = data['value'].at[i] * 17.1037
            except:
                data['value'].at[i] = np.nan
        return data
    
    
    
    # Check if next FiO2 is needed
    def _check_PEEP(self, row: Any, time_table: pd.DataFrame, patient_horowitz_value_list: list, patient_horowitz_abs_time_list: list, patient_horowitz_charttime_list: list, index: int):
        
        # Check if PEEP nearest to fiO2 is > 5
        for j in range(index, 0, -1):
            if not math.isnan(time_table['PEEP'].at[j]) and time_table['PEEP'].at[j]< 5:
                return False
            elif not math.isnan(time_table['PEEP'].at[j]):
                
                # Calculate and store horowitz
                patient_horowitz_value_list.append(row['paO2_(ohne_Temp-Korrektur)']/time_table['fiO2'].at[index])
                patient_horowitz_abs_time_list.append(row['abstime'])
                patient_horowitz_charttime_list.append(row['charttime'])
                return False
        return True


    def _check_ventilated(self, admission, paO2_time, fio2time):
        # Only use data where patient is ventilated
        check_ventilated = False

        for i in range(len(admission[MECHVENTSTART])):
            start_time = admission[MECHVENTSTART][i]
            end_time = admission[MECHVENTEND][i]

            if start_time <= paO2_time <= end_time and start_time <= fio2time <= end_time:
                check_ventilated = True
                break
        return check_ventilated

    def _impute_find_indexes(self, data: pd.DataFrame, value_column: str, time_column: str, cutoff: int, intervall: int, filler) -> pd.DataFrame:
        data = data.sort_values(by=time_column)
        data = data.reset_index()
        del data['index']
        first_index = np.nan
        last_index = np.nan
        for index, row in data.iterrows():
            if math.isnan(first_index) and not math.isnan(row[value_column]):
                first_index = index
                last_index = index
            if not math.isnan(row[value_column]) and last_index<index:
                last_index = index
        if not (math.isnan(first_index) or math.isnan(last_index)) :
            data = self._fill_in_between(data, value_column, time_column, first_index, last_index, cutoff, intervall, filler)
        return data

    def _fill_in_between(self, data: pd.DataFrame, value_column: str, time_column: str, start: int, end: int, cutoff, intervall: int, filler) -> pd.DataFrame:
        last_value = 0
        last_time = np.nan
        times = []
        values = []
        for index in np.arange(start, end, float(intervall)):
            if not math.isnan(data[value_column].at[index]):
                last_value = data[value_column].at[index]
                last_time = data[time_column].at[index]
                times.append(last_time)
                values.append(last_value)
            elif cutoff == -1 or (data[time_column].at[index]-last_time<cutoff):
                times.append(data[time_column].at[index])
                values.append(data[value_column].at[index])
            else :
                times.append(data[time_column].at[index])
                values.append(filler)
        processed_df = pd.DataFrame(columns=[time_column, value_column])
        processed_df[time_column] = times
        processed_df[value_column] = values
        return processed_df
