import math

from typing import Any
from functools import reduce
from datetime import datetime

import pandas as pd
import numpy as np
from numpy import ndarray
from Lookup.MIMICIV_item_dict import *
from Lookup.uka_extraction_indicies import *
from Lookup.static_information import *
from Lookup.features_new import *
from preprocessing.IPreprocessor import IPreprocessor



"""Class that is used to preprpcess Data extracted from the Extractor. The class provides various functions already used during extraction.
The most important function is process_data which takes the extracted data and returns the feature vector"""
class MIMICIVPreprocessor(IPreprocessor):
    def __init__(self, options: Any, job_id: int):
        super().__init__(options, job_id)
        self.options = options

    def process_data_new(self, list_lab_df: list, list_chart_df: list, list_input_df: list, list_procedures_df: list,
                         ventilation: pd.DataFrame,  patient_horowitz: pd.DataFrame, data_vol_input: pd.DataFrame, data_vol_output: pd.DataFrame, admission, times: list, min_horowitz_time_stamp: float) :
        """Function that creates a dataframe which has values on all parameters at all times. If no information is known for a timestamp its value is NaN.
         Information on admission, as well times important for the windows is needed"""
        
        variable_df = self.generate_variable_page()
        variable_df['Zeitpunkt'] = times

        # Process information that is presumed to be mostly unchanged during an admission
        static_information = self._assign_static_information(admission, min_horowitz_time_stamp, other_code_reason)

        # Create dataframe containing all timepoints in the window in 1 minutes intervalls
        times_complete = pd.DataFrame(data=times, columns=['charttime'])

        # Calculate fluid balance
        fluid_balance = self._calculate_fluid_balance(data_vol_input, data_vol_output, times)
        fluid_balance = times_complete.merge(fluid_balance, on='charttime', how = 'left')
        patient_horowitz = times_complete.merge(patient_horowitz[['charttime', 'Horowitz-Quotient_(ohne_Temp-Korrektur)']], on='charttime', how = 'left')
        
        variable_df['Horowitz-Quotient_(ohne_Temp-Korrektur)'] = patient_horowitz['Horowitz-Quotient_(ohne_Temp-Korrektur)']
        variable_df['24h-Bilanz_(Fluessigkeiten-Einfuhr_vs_-Ausfuhr)'] = fluid_balance['24h-Bilanz_(Fluessigkeiten-Einfuhr_vs_-Ausfuhr)']

        # Init variables
        chart_data = []
        chart_ids = []
        lab_data = []
        lab_ids = []
        input_data = []
        input_ids = []
        procedure_data = []
        procedure_ids = []
        calculated = [[] for _ in range(4)]

        # Safe if a parameter that has multiple items has been visited
        multiple_visited = [False for _ in range(len(set(dict_multiple.values())))]

        # Iterate over all chart data
        for i in range(len(chart_index_dict)):
            itemid = chart_index_dict_rev.get(i)

            # Check if itemid has other corresponding itemids and choose fitting preprocessor
            if itemid in list(dict_multiple.keys()) and not multiple_visited[dict_multiple.get(itemid)]:
                multiple_visited[dict_multiple.get(itemid)] = True
                multiple_id = dict_multiple.get(itemid)
                itemids = dict_multiple_rev.get(multiple_id)
                item_indicies = [chart_index_dict.get(item) for item in itemids]
                dataframe_list = [list_chart_df[index] for index in item_indicies]

                chart_data.append(self._process_data_lab_chart_multiple(itemids, dataframe_list, times_complete))
                chart_ids.append(itemids[0])
            
            # Skip Features for which there are multiple items and atleast one was  already used in a previous iteration 
            elif itemid in list(dict_multiple.keys()):
                continue

            # Skip Expiration and Inspiration Ratio as they are used to calculated I:E
            elif itemid == 226871 or itemid == 226873:
                continue
            else:
                # Process data with only one type of measurement collected in the Database
                chart_data.append(self._process_data_lab_chart(itemid, list_chart_df[i], ventilation,
                                                               list_chart_df[chart_index_dict.get(229267)],
                                                               times_complete))
                chart_ids.append(itemid)

        
        # Calculated parameters not readily available
        calculated[0] = self._calculate_ind_Tidal_Volume(list_chart_df[chart_index_dict.get(224685)], static_information[static_dict.get('Geschlecht')], 
                                                         static_information[static_dict.get('Groesse')], times_complete)

        calculated[1] = self._calculate_Lymphs_percent(list_chart_df[chart_index_dict.get(229358)],
                                                       list_lab_df[lab_index_dict.get(51301)], times_complete)

        calculated[2] = self._calculate_deltaP(list_chart_df[chart_index_dict.get(220339)],
                                               list_chart_df[chart_index_dict.get(224696)], times_complete)

        calculated[3] = self._calculate_IE(list_chart_df[chart_index_dict.get(226871)],
                                           list_chart_df[chart_index_dict.get(226873)], times_complete)

        # Process lab related parameters
        for i in range(len(lab_index_dict)):
            itemid = lab_index_dict_rev.get(i)

            # Process lab parameters with multiple types of measurement
            if itemid in list(dict_multiple.keys()) and not multiple_visited[dict_multiple.get(itemid)]:
                multiple_visited[dict_multiple.get(itemid)] = True
                multiple_id = dict_multiple.get(itemid)
                itemids = dict_multiple_rev.get(multiple_id)

                # Get all relevant type of measurements
                item_indicies = [lab_index_dict.get(item) for item in itemids]
                dataframe_list = [list_lab_df[index] for index in item_indicies]

                lab_data.append(self._process_data_lab_chart_multiple(itemids, dataframe_list, times_complete))
                lab_ids.append(itemids[0])

            # Skip type of measurements that already have been processed
            elif itemid in list(dict_multiple.keys()):
                continue
            
            # Process parameters with only a single type of measurement
            else:
                lab_data.append(self._process_data_lab_chart(itemid, list_lab_df[i], ventilation,
                                                             list_chart_df[chart_index_dict.get(229267)],
                                                             times_complete))
                lab_ids.append(itemid)

        # Process all IV drug inputs
        for i in range(len(input_index_dict)):
            # We dont have multiple values for a input feature therefore we omit this part here
            itemid = input_index_dict_rev.get(i)
            input_data.append(self._process_data_input(itemid, list_input_df[i], times_complete))
            input_ids.append(itemid)


        # I presume procedure data is empty since my querries dont seem to find anything
        if len(list_procedures_df[0].index) > 0 or len(list_procedures_df[1].index) > 0:
            print("Procedure found")
            print(len(list_procedures_df))

        # Create Complete DF for all parameters at all times
        variable_df = self._assign_variables(chart_data, chart_ids, lab_data, lab_ids, input_data, input_ids,
                                             procedure_data, procedure_ids, calculated, variable_df)
        variable_df = variable_df.sort_values(by='Zeitpunkt')

        # Create feature vector and calculate frequency of each parameter
        feature_df, freq_df = self._assign_features(variable_df, static_information, feature_position_dict_eICU_MIMIC)
        return  feature_df, freq_df 

        

    def _process_data_input(self, itemid: int, data: pd.DataFrame, times_complete: pd.DataFrame) -> pd.DataFrame:
        """This function process all IV drug inputs"""

        # Choose the right processing function
        data = data.fillna(value=np.nan)
        if itemid == 221794:
            processed = self._process_Furosemid(data)
        elif itemid == 221906:
            processed = self._process_Norepinephrin(data)
        elif itemid == 222168:
            processed = self._process_Propofol(data)
        elif itemid == 222315:
            processed = self._process_Vasopressin(data)
        elif itemid == 221668:
            processed = self._process_Midazolam(data)
        elif itemid == 225154:
            processed = self._process_Morphin(data)
        elif itemid == 221986:
            processed = self._process_Milrinon(data)
        elif itemid == 221744:
            processed = self._process_Fentanyl(data)
        elif itemid == 221653:
            processed = self._process_Dobutamin(data)
        elif itemid == 221712:
            processed = self._process_Ketanest(data)
        elif itemid == 221289:
            processed = self._process_Epinephrin(data)
        elif itemid == 229420:
            processed = self._process_Dexmedetomidin(data)
        elif itemid == 229233:
            processed = self._process_Rocuronium(data)
        else:
            print("Unsuitable input id: " + str(itemid))

        # Converted med administration to rate at each timestamp
        processed = self._process_meds_converted(processed, times_complete)
        return processed

    def _process_data_procedures(self, itemdid: int, data: pd.DataFrame, times_complete: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(columns=["charttime", "value"])


    def _process_data_lab_chart(self, itemid: int, data: pd.DataFrame, ventilation: pd.DataFrame,
                                ecmo_present: pd.DataFrame, times_complete: pd.DataFrame) -> pd.DataFrame:
        """Process all features either in lab or chart data"""
        data = data.fillna(value=np.nan)

        # Check which chart features need to be filled
        data = data.sort_values(by='charttime')
        data = data.reset_index()
        del data['index']
        if itemid == 50862:
            data = self._albumin(data)
        elif itemid == 50912:
            data = self._creatinine(data)
        elif itemid == 51006:
            data = self._urea_nitrogen(data)
        elif itemid == 228640:
            data = self._etCO2(data)
        elif itemid == 50889:
            data = self._CRP(data, 9.5238)

        elif itemid == chart_item_dict.get('PEEP'):
            data = self._process_Ventilation_features(data, ventilation)
        elif itemid == 223835:
            data = self._process_Ventilation_features(data, ventilation)

        elif itemid == chart_item_dict.get('P_EI'):
            data = self._process_Ventilation_features(data, ventilation)
        elif itemid == 224093:
            data = self._process_Position(data)

        elif itemid == 229278:
            data = self._process_ECMO_Gas_Flow(data, ecmo_present)
        elif itemid == 229280:
            data = self._process_ECMO_FiO2(data, ecmo_present)
        elif itemid == 229270:
            data = self._process_ECMO_Blood_Flow(data, ecmo_present)

        data = data.merge(times_complete, how='right', on='charttime')
        return data

    # Handle features with multiple itemids
    def _process_data_lab_chart_multiple(self, itemids: list, data: list, times_complete: pd.DataFrame):
        """This function is used for processing parameters that are stored in different entries in the database"""
        
        data_filled = []
        
        # Choose if processing is needed and apply it
        for df in data:
            data_filled.append(df.fillna(value=np.nan))
        if 223761 in itemids and 223762 in itemids and len(itemids) == 2:
            data_processed = self._process_Temperature(data_filled)
        elif 227547 in itemids and 228374 in itemids and len(itemids) == 2:
            data_processed = self._process_Stroke_volume(data_filled)
        elif 51222 in itemids and 50811 in itemids and len(itemids) == 2:
            data_processed = self._process_Hemoglobin(data_filled)
        elif 224690 in itemids and 220210 in itemids and len(itemids) == 2:
            data_processed = self._process_RR(data_filled)
        elif 224689 in itemids and 224422 in itemids and len(itemids) == 2:
            data_processed = self._process_RR_spont(data_filled)
        elif 224686 in itemids and 224421 in itemids and len(itemids) == 2:
            data_processed = self._process_Vt_spont(data_filled)




        else:
            print("No suitable ids for multiple")
            data_processed = pd.DataFrame(columns=['charttime', 'value'])

        data_processed = times_complete.merge(data_processed, on='charttime', how='left')
        return data_processed

    def _assign_variables(self, chart_data: list, chart_ids: list, lab_data: list, lab_ids: list, input_data: list,
                          input_ids: list,
                          procedure_data: list, procedure_ids: list, calculated: list,
                          variable_df: pd.DataFrame) -> pd.DataFrame:
        """This function is used to assign the different parameters to their respective column in a dataframe that is the same throughout all databases"""

        # Assign the data for each parameter to their respective id 
        # First find the index where the data is stored and than find the column in the variable df
        for index, itemid in enumerate(chart_ids):
            column = mimiciv_mapping.get(itemid)
            data = chart_data[index]
            variable_df[column] = data['value']

        for index, itemid in enumerate(lab_ids):
            column = mimiciv_mapping.get(itemid)
            data = lab_data[index]
            variable_df[column] = data['value']

        for index, itemid in enumerate(input_ids):
            column = mimiciv_mapping.get(itemid)
            data = input_data[index]
            variable_df[column] = data['value']

        for index, itemid in enumerate(procedure_ids):
            column = mimiciv_mapping.get(itemid)
            data = procedure_data[index]
            variable_df[column] = data['value']

        # Asssign the variables that had to be calculated
        variable_df['individuelles_Tidalvolumen_pro_kg_idealem_Koerpergewicht'] = calculated[0]['value']
        variable_df['Lymphozyten_prozentual'] = calculated[1]['value']
        variable_df['deltaP'] = calculated[2]['value']
        variable_df['I:E'] = calculated[3]['value']

        return variable_df

    def _calculate_horowitz(self, admissions: ndarray, id_dict: dict, list_abs_times_PEEP: list, list_times_PEEP: list,
                            list_values_PEEP: list,
                            list_abs_times_fiO2: list, list_values_fiO2: list, list_times_fiO2: list,
                            list_abs_times_paO2: list, list_values_paO2: list,
                            list_times_paO2: list, ) -> tuple:

        """Function that calculates the horovitz-index. Information on admissions, PEEP, FiO2 and paO2 is needed"""

        # Save all horowitz for patients
        patients_horowitz_list = []
        patients_min_horowitz_list = []

        # Iterate over all admissions
        for admission in admissions:

            # Get admission id to extract correct data from list
            admission_id = int(str(admission[ADMISSION_ID])[1:-1])
            index_list = id_dict.get(admission_id)

            # Create dataframe for PEEP
            df_PEEP = pd.DataFrame(columns=['abstime', 'charttime', "PEEP"])
            df_PEEP.set_index('abstime')
            df_PEEP['abstime'] = list_abs_times_PEEP[index_list]
            df_PEEP['PEEP'] = list_values_PEEP[index_list]
            df_PEEP['charttime'] = list_times_PEEP[index_list]

            # Create dataframe for fiO2
            df_fiO2 = pd.DataFrame(columns=['abstime', 'charttime', "fiO2"])
            df_fiO2.set_index('abstime')
            df_fiO2['abstime'] = list_abs_times_fiO2[index_list]
            df_fiO2['fiO2'] = list_values_fiO2[index_list]
            df_fiO2['charttime'] = list_times_fiO2[index_list]

            # Create dataframe for paO2
            df_paO2 = pd.DataFrame(columns=['abstime', 'charttime', "paO2_(ohne_Temp-Korrektur)"])
            df_paO2.set_index('abstime')
            df_paO2['abstime'] = list_abs_times_paO2[index_list]
            df_paO2['paO2_(ohne_Temp-Korrektur)'] = list_values_paO2[index_list]
            df_paO2['charttime'] = list_times_paO2[index_list]

            # Merge data_frames into one
            data_frames = [df_PEEP, df_paO2, df_fiO2]
            time_table = reduce(lambda left, right: pd.merge(left, right, on=['abstime', 'charttime'], how='outer'),
                                data_frames)
            time_table.sort_values(by="abstime", inplace=True)
            time_table.reset_index(drop=True, inplace=True)

            # Calculate Horowitz where PEEP is at least 5
            patient_horowitz_value_list = []
            patient_horowitz_abs_time_list = []
            patient_horowitz_charttime_list = []
            
            # Search for non NaN paO2 values
            for index, row in time_table.iterrows():
                
                if not math.isnan(row['paO2_(ohne_Temp-Korrektur)']):
                    # Search for next fiO2 recorded before the paO2 value
                    for i in range(index, 0, -1):
                        if not math.isnan(time_table['fiO2'].at[i]) and not time_table['fiO2'].at[i] == 0:
                            
                            # Ensure a patient is ventilated because over wiese horovitz makes no sense
                            if not self._check_ventilated(admission, row['charttime'], time_table['charttime'].at[i]):
                                break

                            # Check if PEEP for found timestamp is higher than 5cm H20
                            if not self._check_PEEP(row, time_table, patient_horowitz_value_list,
                                                    patient_horowitz_abs_time_list, patient_horowitz_charttime_list, i):
                                break

            # Check if atleast one horovitz value is present
            if not len(patient_horowitz_value_list) == 0:
                admission_id_list = [admission_id for _ in range(len(patient_horowitz_abs_time_list))]
                patient_horowitz_df = pd.DataFrame(
                    columns=['abstime', 'charttime', 'admission_id', 'Horowitz-Quotient_(ohne_Temp-Korrektur)'])

                # Create dataframe for horovitz data
                patient_horowitz_df['admission_id'].astype(int)
                patient_horowitz_df['abstime'] = patient_horowitz_abs_time_list
                patient_horowitz_df['charttime'] = patient_horowitz_charttime_list

                patient_horowitz_df['admission_id'] = admission_id_list
                patient_horowitz_df['Horowitz-Quotient_(ohne_Temp-Korrektur)'] = patient_horowitz_value_list

                # Find time of minimum horovitz-index
                min_horowitz_abs_time, min_horowitz_charttime = self._find_lowest_Horowitz(patient_horowitz_df)
                
                # Create dataframe containing admission id and time of minimal horovitz
                patient_min_horowitz_time_df = pd.DataFrame(columns=['admission_id', 'abstime', 'charttime'])
                patient_min_horowitz_time_df['admission_id'] = [admission_id]
                patient_min_horowitz_time_df['abstime'] = [min_horowitz_abs_time]
                patient_min_horowitz_time_df['charttime'] = [min_horowitz_charttime]
                
                # Add data to list to be able to return one df for all admissions
                patients_min_horowitz_list.append(patient_min_horowitz_time_df)
                patients_horowitz_list.append(patient_horowitz_df)
        
        # Concat all admission dataframes into one big dataframe
        if not (len(patients_min_horowitz_list) == 0 or len(patients_horowitz_list) == 0):
            patients_horowitz_df = pd.concat(patients_horowitz_list)
            patients_horowitz_min_time_df = pd.concat(patients_min_horowitz_list)
        else:
            patients_horowitz_df = pd.DataFrame(
                columns=['abstime', 'charttime', 'admission_id', 'Horowitz-Quotient_(ohne_Temp-Korrektur)'])
            patients_horowitz_min_time_df = pd.DataFrame(columns=['admission_id', 'abstime', 'charttime'])

        return patients_horowitz_df, patients_horowitz_min_time_df

    def _find_lowest_Horowitz(self, patient_horowitz: pd.DataFrame):
        """Function that finds the timestamp of lowest horovitz"""

        # Find timestamp for minimal horovitz index and return it
        idx_min_avg_horowitz, min_horowitz = super().find_lowest_Horowitz(patient_horowitz)
        min_horowitz_abs_time = patient_horowitz['abstime'][idx_min_avg_horowitz]
        min_horowitz_charttime = patient_horowitz['charttime'][idx_min_avg_horowitz]
        return min_horowitz_abs_time, min_horowitz_charttime

    
    def _check_PEEP(self, row: Any, time_table: pd.DataFrame, patient_horowitz_value_list: list,
                    patient_horowitz_abs_time_list: list, patient_horowitz_charttime_list: list, index: int):
        """Function that checks if the PEEP next to Choosen Fio2 and paO2 is atleast 5cmH20"""
        # Check if PEEP nearest to fiO2 is > 5
        for j in range(index, 0, -1):
            if not math.isnan(time_table['PEEP'].at[j]) and time_table['PEEP'].at[j] < 5:
                return False
            elif not math.isnan(time_table['PEEP'].at[j]):

                # Calculate and store horowitz
                patient_horowitz_value_list.append(row['paO2_(ohne_Temp-Korrektur)'] / time_table['fiO2'].at[index])
                patient_horowitz_abs_time_list.append(row['abstime'])
                patient_horowitz_charttime_list.append(row['charttime'])
                return False
        return True

    

    # Processing of single features

    
    def _process_Position(self, data: pd.DataFrame) -> pd.DataFrame:
        """This function is used to process a patients position. Because only changes in position are recorded the position gets imputed between changes"""
        
        # Check if at least one position is stored
        if len(data.index) == 0:
            data_processed = pd.DataFrame(columns=['charttime', 'value'])
        else:

            # Find first and last index and charttime of recorded positions
            final_index = np.nan
            first_index = np.nan
            first_charttime = np.nan
            last_charttime = np.nan
            for index, row in data.iterrows():
                if not math.isnan(row['patientpos']) and math.isnan(first_index):
                    first_index = index
                    first_charttime = row['charttime']
                if not math.isnan(row['patientpos']):
                    final_index = index
                    last_charttime = row['charttime']
            
            # Create DF for every minute between first and last charttime for recorded position
            data = data.set_index('charttime')
            data_processed = pd.DataFrame(columns=['charttime', 'value'])
            data_processed['charttime'] = [i for i in np.arange(first_charttime, last_charttime, 1.0)]
            data_processed = data_processed.set_index('charttime')
            
            # For each minute fill in known information about position
            for index, row in data.iterrows():
                if not math.isnan(row['patientpos']):
                    data_processed['value'].at[index] = row['patientpos']
            
            # Impute positions between changes
            last_value = np.nan
            value_list = []
            for index, row in data_processed.iterrows():
                if math.isnan(row['value']) and index > first_charttime and index <= last_charttime:
                    value_list.append(last_value)
                elif index >= first_charttime and index < last_charttime and not math.isnan(row['value']):
                    last_value = row['value']
                    value_list.append(last_value)
                else:
                    value_list.append(np.nan)
            data_processed['value'] = value_list
        return data_processed

    def process_implied_Ventilation(self, times: list) -> tuple:
        """This function is used to determine possible ventilation timeframes by using variables that are present during ventilation"""
        
        times_set = set(times)
        splits = []
        start_time = []
        end_time = []
        duration = []
        # Allow for a maximum of 5 hours between occurence of records that imply ventilation before starting a new ventalition timeframe
        max_time_dif = 300

        # Set artifical nan to compare to
        nan_string = "0001-01-01 00:00:00"
        nan = datetime.strptime(nan_string, '%Y-%m-%d %H:%M:%S')
        first_time = nan
        last_time = nan

        # Go through every time that has a ventilation implied and create timeframes where the patient is most likely ventilated
        for time in times_set:
            if first_time == nan:
                first_time = time
                last_time = time
                continue
            else:
                time_dif = (time - last_time).total_seconds() / 60.0
                # Only create new timeframe if the difference in time between two values is greater than 5h
                if time_dif > max_time_dif:
                    splits.append((first_time, last_time))
                    first_time = time
                last_time = time
        
        # Create lists that contain start-/endtime and duration of ventilation and return them
        for split in splits:
            start_time.append(split[0])
            end_time.append(split[1])
            duration.append((split[1] - split[0]).total_seconds() / 60.0)
        return (start_time, end_time, duration)


    def _calculate_IE(self, exp: pd.DataFrame, insp: pd.DataFrame, times_complete: pd.DataFrame) -> pd.DataFrame:
        """This function calculates the Inspiratory Expiratory Ratio"""

        # Init variable and rename colums for easier access
        inspiration = pd.DataFrame(insp)
        expiration = pd.DataFrame(exp)
        inspiration = inspiration.rename(columns={'value': 'insp ratio'})
        expiration = expiration.rename(columns={'value': 'exp ratio'})

        # Merge Inspiratory and Expiratory data
        ie_data = inspiration.merge(expiration, how="outer", on='charttime')
        ie_data = ie_data.sort_values(by='charttime')
        ie_values = []

        # Calculate the I:E for every entry store nan if not possible Normale Insp Ratio and Exp ratio should be available at the same time
        for _, row in ie_data.iterrows():

            try:
                value = row['insp ratio'] / row['exp ratio']
            except:
                value = math.nan
            ie_values.append(value)
        
        # Create new dataframe
        ie_data['value'] = ie_values
        del ie_data['insp ratio']
        del ie_data['exp ratio']
        ie_data = ie_data.merge(times_complete, on='charttime', how='right')
        ie_data = ie_data.sort_values(by='charttime')
        ie_data = ie_data.reset_index()
        del ie_data['index']
        return ie_data

    def _calculate_ind_Tidal_Volume(self, data: pd.DataFrame, gender: str, height: float,
                                    times_complete: pd.DataFrame) -> pd.DataFrame:
        """ This function calculates the individual tidalvolume per kg of ideal bodyweight"""
        # Create dataframe for ind VT
        tidal_volume = pd.DataFrame(columns=['charttime', 'value'])
        
        # Calculate ideal bodyweight, defaults to calculation for men
        if gender == 'M':
            ideal_body_weight = 50 + (0.91 * ((height) - 152.4))
        elif gender == 'F':
            ideal_body_weight = 45.59 + (0.91 * ((height) - 152.4))
        else:
            ideal_body_weight = 50 + (0.91 * ((height) - 152.4))
        ind_tv = []
        ind_tv_times = []
        
        # Calculate ind_Tidal_Volume for every entry of a tidal volume and store the timestamp
        for _, row in data.iterrows():
            ind_tv.append(row['value'] / ideal_body_weight)
            ind_tv_times.append(row['charttime'])
        
        # Create dataframe
        tidal_volume['value'] = ind_tv
        tidal_volume['charttime'] = ind_tv_times
        tidal_volume = tidal_volume.merge(times_complete, on='charttime', how='right')
        tidal_volume = tidal_volume.sort_values(by='charttime')
        tidal_volume = tidal_volume.reset_index()
        del tidal_volume['index']
        return tidal_volume

    
    def _calculate_Lymphs_percent(self, lymphs: pd.DataFrame, leukos: pd.DataFrame,
                                  times_complete: pd.DataFrame) -> pd.DataFrame:
        """This function calculates the percentage of lymphocytes in the blood"""

        # Rename columns for easier access and merged data for absolute lymphocyte count and leucocyte count
        lymphs = lymphs.rename(columns={'value': 'lymphs'})
        leukos = leukos.rename(columns={'value': 'leukos'})
        merged = lymphs.merge(leukos, on='charttime', how='outer')

        percent_values = []
        percent_times = []

        # Calulate percentage of lymphocytes in the blood if data for both absolute lymphocyte count and leucocyte count
        # is available in a specified timeframe 
        for index, row in merged.iterrows():
            if not math.isnan(row['lymphs']):
                lymphs = row['lymphs']
                time = row['charttime']
                for i in range(index, 0, -1):
                    if not math.isnan(merged['leukos'].at[i]) and (time - merged['charttime'].at[i]) > \
                            self.options["extraction_parameter"]["cutoff_time_lymphs_leukos"] and not lymphs == 0:
                        percent_times.append(time)
                        percent_values.append(lymphs / merged['leukos'].at[i])
        
        # Create dataframe      
        lymphs_percent = pd.DataFrame(columns=['charttime', 'value'])
        lymphs_percent['charttime'] = percent_times
        lymphs_percent['value'] = percent_values
        lymphs_percent = lymphs_percent.merge(times_complete, on='charttime', how='right')
        lymphs_percent = lymphs_percent.sort_values(by='charttime')
        lymphs_percent = lymphs_percent.reset_index()
        del lymphs_percent['index']
        return lymphs_percent

    
    def _calculate_deltaP(self, PEEP: pd.DataFrame, PEI: pd.DataFrame, times_complete: pd.DataFrame) -> pd.DataFrame:
        """This function calculates difference between PEEP and PEI"""
        
        # Rename columns for easier access and merge data for PEEP and PEI
        PEEP = PEEP.rename(columns={'value': 'PEEP'})
        PEI = PEI.rename(columns={'value': 'PEI'})
        merged = PEEP.merge(PEI, how='outer', on='charttime')
        merged = merged.sort_values(by='charttime')
        merged = merged.reset_index()
        del merged['index']

        # Calculate deltap
        deltap_times, deltap_values = super()._calculate_deltaP(merged, 'PEEP', 'PEI', 'charttime')

        # Create Dataframe
        deltap = pd.DataFrame(columns=['charttime', 'value'])
        deltap['charttime'] = deltap_times
        deltap['value'] = deltap_values
        return deltap

    # Calculate fluid balance in a 24h window
    def _calculate_fluid_balance(self, input_vol: pd.DataFrame, output: pd.DataFrame,
                                 times: list) -> pd.DataFrame:
        """Function that calculates the fluid balance for one day"""
        
        # Make sure there are no entries spanning the border of 24h by splitting them
        length_of_window = times[-1]  - times[0] 
        entries_to_add = []
        last_i = 0
        for i in range(1440, int(length_of_window), 1440):

            for j in range(len(input_vol)):
                # Make sure Bolus and Drug push are not splitted, since they happen instant
                if input_vol['starttime'].at[j] > last_i and input_vol['starttime'].at[j] < i and \
                        input_vol['endtime'].at[j] > i and (
                        input_vol['way'].at[j] == 'Bolus' or input_vol['way'].at[j] == 'Drug Push'):

                    df_to_add = pd.DataFrame(columns=['admission_id', 'value', 'starttime', 'endtime', 'way'])
                    df_to_add['admission_id'] = [input_vol['admission_id'].at[j]]
                    df_to_add['value'] = [input_vol['value'].at[j]]
                    df_to_add['starttime'] = [input_vol['starttime'].at[j]]
                    df_to_add['endtime'] = [input_vol['starttime'].at[j]]
                    df_to_add['way'] = [input_vol['way'].at[j]]

                    entries_to_add.append(df_to_add)

                # Split medications that cross the border of days into two parts, so that we use the correct amount delivered
                elif input_vol['starttime'].at[j] > last_i and input_vol['starttime'].at[j] < i and \
                        input_vol['endtime'].at[j] > i:

                    # Calculate portion of fluid for each of the two days and split it accordingly
                    time_running = input_vol['starttime'].at[j] - input_vol['endtime'].at[j]
                    time_considered = input_vol['starttime'].at[j] - i
                    value_considered = input_vol['value'].at[j] * time_considered / time_running
                    value_left = input_vol['value'].at[j] - value_considered
                    df_to_add_after = pd.DataFrame(columns=['admission_id', 'value', 'starttime', 'endtime', 'way'])

                    # Add fluid data before day change
                    df_to_add_before = pd.DataFrame(columns=['admission_id', 'value', 'starttime', 'endtime', 'way'])
                    df_to_add_before['admission_id'] = [input_vol['admission_id'].at[j]]
                    df_to_add_before['value'] = [value_considered]
                    df_to_add_before['starttime'] = [input_vol['starttime'].at[j]]
                    df_to_add_before['endtime'] = [i - 1]
                    df_to_add_before['way'] = [input_vol['way'].at[j]]
                    entries_to_add.append(df_to_add_before)

                    # Add fluid data after day change
                    df_to_add_after['admission_id'] = [input_vol['admission_id'].at[j]]
                    df_to_add_after['value'] = [value_left]
                    df_to_add_after['starttime'] = [i]
                    df_to_add_after['endtime'] = [input_vol['endtime'].at[j]]
                    df_to_add_after['way'] = [input_vol['way'].at[j]]

                    entries_to_add.append(df_to_add_after)

                # Store input that happens during one day
                elif input_vol['starttime'].at[j] > last_i and input_vol['starttime'].at[j] < i and \
                        input_vol['endtime'].at[j] < i:
                    df_to_add = pd.DataFrame(columns=['admission_id', 'value', 'starttime', 'endtime', 'way'])
                    df_to_add['admission_id'] = [input_vol['admission_id'].at[j]]
                    df_to_add['value'] = [input_vol['value'].at[j]]
                    df_to_add['starttime'] = [input_vol['starttime'].at[j]]
                    df_to_add['endtime'] = [input_vol['endtime'].at[j]]
                    df_to_add['way'] = [input_vol['way'].at[j]]

                    entries_to_add.append(df_to_add)
            last_i = i

        # Make sure there is no more than one entry for each timestamp by summing up input values for the same timestamp
        if len(entries_to_add) == 0:
            input_vol_cleaned = pd.DataFrame(columns=['admission_id', 'value', 'starttime', 'endtime', 'way'])
            del input_vol_cleaned['endtime']
            del input_vol_cleaned['admission_id']
        else:
            input_vol_cleaned = pd.concat(entries_to_add, ignore_index=True)
            del input_vol_cleaned['endtime']
            del input_vol_cleaned['admission_id']
            input_vol_cleaned = input_vol_cleaned.groupby('starttime').sum()


        del output['admission_id']
        output_cleaned = output.groupby('charttime').sum()

        # Make sure there is an entry for every time during the admission
        full_times_df = pd.DataFrame(columns=['charttime'], data=times)
        full_output = full_times_df.merge(output_cleaned, how='outer', left_on='charttime', right_on='charttime')
        full_input = full_times_df.merge(input_vol_cleaned, how='outer', left_on='charttime', right_on='starttime')

        if not 'value' in full_input.columns:
            print("error")
        # Calculate input and output values for each day if no input or output is recorded store nan
        day_input = full_input['value'].groupby(np.arange(len(full_input['value'])) // 1440).sum(min_count=1)
        day_output = full_output['value'].groupby(np.arange(len(full_output['value'])) // 1440).sum(min_count=1)

        # Create with rows for each day
        time = full_input['charttime'].groupby(np.arange(len(full_input['charttime'])) // 1440).max()

        # Calculate difference between input and output for each day
        balance_list = []

        for i in range(day_input.size):
            if not (math.isnan(day_input.at[i]) or math.isnan(day_output.at[i])):
                balance_list.append(day_input.at[i] - day_output.at[i])
            elif not math.isnan(day_output.at[i]):
                balance_list.append(-day_output.at[i])
            elif not math.isnan(day_input.at[i]):
                balance_list.append(day_input.at[i])
            else:
                balance_list.append(np.nan)
        balance = pd.DataFrame(columns=['charttime', '24h-Bilanz_(Fluessigkeiten-Einfuhr_vs_-Ausfuhr)'])

        # Add data to df
        balance['charttime'] = time
        balance['24h-Bilanz_(Fluessigkeiten-Einfuhr_vs_-Ausfuhr)'] = balance_list
        return balance

    # Processing of features with are represented more than once in the data
    
    def _process_Temperature(self, data: list) -> pd.DataFrame:
        """Process bodytemperature in both °F and °C and converts it to °C, if multiple temperatures are present at the same
        timestamp the average is taken"""

        # Create DF for both measurement units
        temp_C = pd.DataFrame(columns=['charttime', 'C'])
        temp_F = pd.DataFrame(columns=['charttime', 'F'])

        # Add every entry in the data to their respective dataframe according to their unit
        for i in range(len(data)):
            if (len(data[i]['itemid'])) > 0:
                match data[i]['itemid'].at[0]:
                    case 223761:
                        temp_F['charttime'] = data[i]['charttime']
                        # Convert F to C
                        temp_F['F'] = self._fahrenheit(data[i])['value']
                    case 223762:
                        temp_C['charttime'] = data[i]['charttime']
                        temp_C['C'] = data[i]['value']

        # Merge converted temperatures into one dataframe
        merged_temps = temp_C.merge(temp_F, on='charttime', how='outer')
        temps = []

        # Calculate average as needed otherwise just store available temperature
        for _, row in merged_temps.iterrows():
            if not math.isnan(row['C']) and not math.isnan(row['F']):
                temps.append((row['C'] + row['F']) / 2)
            elif not math.isnan(row['C']):
                temps.append(row['C'])
            elif not math.isnan(row['F']):
                temps.append(row['F'])
            else:
                temps.append(np.nan)

        # Create final DF
        del merged_temps['F']
        del merged_temps['C']
        merged_temps['value'] = temps

        return merged_temps

    def _process_Stroke_volume(self, data: list) -> pd.DataFrame:
        """Function that process the stroke volume and returns an DF containing all data"""
        
        # Initialize DFs for both measurement types
        stroke_vol_1 = pd.DataFrame(columns=['charttime', 'vol1'])
        stroke_vol_2 = pd.DataFrame(columns=['charttime', 'vol2'])

        # Add every entry in the data to their respective dataframe according itemid
        for i in range(len(data)):
            if (len(data[i]['itemid'])) > 0:
                match data[i]['itemid'].at[0]:
                    case 228374:
                        stroke_vol_1['charttime'] = data[i]['charttime']
                        stroke_vol_1['vol1'] = data[i]['value']
                    case 227547:
                        stroke_vol_2['charttime'] = data[i]['charttime']
                        stroke_vol_2['vol2'] = data[i]['value']
        
        # Merge stroke volumes into one dataframe
        merged_vols = stroke_vol_1.merge(stroke_vol_2, on='charttime', how='outer')
        vols = []
        
        # Calucalte average as needed otherwise just store available stroke volume
        for _, row in merged_vols.iterrows():
            if not math.isnan(row['vol1']) and not math.isnan(row['vol2']):
                vols.append((row['vol1'] + row['vol2']) / 2)
            elif not math.isnan(row['vol1']):
                vols.append(row['vol1'])
            elif not math.isnan(row['vol2']):
                vols.append(row['vol2'])
            else:
                vols.append(np.nan)

        # Create Final DF
        del merged_vols['vol1']
        del merged_vols['vol2']
        merged_vols['value'] = vols

        return merged_vols

    def _process_Hemoglobin(self, data: list) -> pd.DataFrame:
        """Function that process the Hemoglobin values and returns a single DF containing all values with their respective timestamp"""
        
        # Initialize DFs for both measurements
        hemo_1 = pd.DataFrame(columns=['charttime', 'hemo1'])
        hemo_2 = pd.DataFrame(columns=['charttime', 'hemo2'])
        
        # Add every entry in the data to their respective dataframe according itemid
        for i in range(len(data)):
            if (len(data[i]['itemid'])) > 0:
                match data[i]['itemid'].at[0]:
                    case 51222:
                        hemo_1['charttime'] = data[i]['charttime']

                        # Convert to right unit
                        hemo_1['hemo1'] = self._Hemoglobin(data[i])['value']
                    case 50811:
                        hemo_2['charttime'] = data[i]['charttime']

                        # Convert to right unit
                        hemo_2['hemo2'] = self._Hemoglobin(data[i])['value']

        # Merge values into one dataframe
        merged_hemo = hemo_1.merge(hemo_2, on='charttime', how='outer')
        hemos = []

        # Calucalte average as needed otherwise just store available value
        for _, row in merged_hemo.iterrows():
            if not math.isnan(row['hemo1']) and not math.isnan(row['hemo2']):
                hemos.append((row['hemo1'] + row['hemo2']) / 2)
            elif not math.isnan(row['hemo1']):
                hemos.append(row['hemo1'])
            elif not math.isnan(row['hemo2']):
                hemos.append(row['hemo2'])
            else:
                hemos.append(np.nan)

        # Create Final DF
        del merged_hemo['hemo1']
        del merged_hemo['hemo2']
        merged_hemo['value'] = hemos
        return merged_hemo

    def _process_RR(self, data: list) -> pd.DataFrame:
        """Function that processes the Respiratory Rate and return a single DF containing all values and their timestamps"""

        # Initialize DF for both measurements
        RespRate1 = pd.DataFrame(columns=['charttime', 'total'])
        RespRate2 = pd.DataFrame(columns=['charttime', 'RR'])

        # Add every entry in the data to their respective dataframe according itemid
        for i in range(len(data)):
            if (len(data[i]['itemid'])) > 0:
                match data[i]['itemid'].at[0]:
                    case 224690:
                        RespRate1['charttime'] = data[i]['charttime']
                        RespRate1['total'] = data[i]['value']
                    case 220210:
                        RespRate2['charttime'] = data[i]['charttime']
                        RespRate2['RR'] = data[i]['value']
        
        # Merge values into one dataframe
        merged_RR = RespRate1.merge(RespRate2, on='charttime', how='outer')
        values = []

        # Add values for Respiratory Rate in list, prioritize TOTAL RR over RR
        for _, row in merged_RR.iterrows():
            if not math.isnan(row['total']):
                values.append(row['total'])
            elif not math.isnan(row['RR']):
                values.append(row['RR'])
            else:
                values.append(np.nan)
        merged_RR['value'] = values
        return merged_RR[['charttime', 'value']]

    def _process_RR_spont(self, data: list) -> pd.DataFrame:
        """Function that processes the spontanous Respiratory Rate and return a single DF containing all values and their timestamps"""
        
        # Initialize DFs for both measurements
        RespRate1 = pd.DataFrame(columns=['charttime', 'Spont RR'])
        RespRate2 = pd.DataFrame(columns=['charttime', 'RR spont'])

        # Add every entry in the data to their respective dataframe according itemid
        for i in range(len(data)):
            if (len(data[i]['itemid'])) > 0:
                match data[i]['itemid'].at[0]:
                    case 224422:
                        RespRate1['charttime'] = data[i]['charttime']
                        RespRate1['Spont RR'] = data[i]['value']
                    case 224689:
                        RespRate2['charttime'] = data[i]['charttime']
                        RespRate2['RR spont'] = data[i]['value']
        
        #  Merge values into one dataframe
        merged_RR = RespRate1.merge(RespRate2, on='charttime', how='outer')
        values = []
        
        # Calucalte average as needed otherwise just store available value
        for _, row in merged_RR.iterrows():
            if not math.isnan(row['RR spont']) and not math.isnan(row['Spont RR']):
                values.append((row['RR spont'] + row['Spont RR'] / 2))
            elif not math.isnan(row['Spont RR']):
                values.append(row['Spont RR'])
            elif not math.isnan(row['RR spont']):
                values.append(row['RR spont'])
            else:
                values.append(np.nan)
        merged_RR['value'] = values
        return merged_RR[['charttime', 'value']]

    def _process_Vt_spont(self, data: list) -> pd.DataFrame:
        """Function that processes the spotanous tidal volum and returns a DF with all values and their timestamps"""
        
        # Initialize df for both measurements
        Vt_spont1 = pd.DataFrame(columns=['charttime', 'Spont Vt'])
        Vt_spont2 = pd.DataFrame(columns=['charttime', 'TV spont'])

        # Add every entry in the data to their respective dataframe according itemid
        for i in range(len(data)):
            if (len(data[i]['itemid'])) > 0:
                match data[i]['itemid'].at[0]:
                    case 224421:
                        Vt_spont1['charttime'] = data[i]['charttime']
                        Vt_spont1['Spont Vt'] = data[i]['value']
                    case 224686:
                        Vt_spont2['charttime'] = data[i]['charttime']
                        Vt_spont2['TV spont'] = data[i]['value']
        
        #  Merge values into one dataframe
        merged_TV_spont = Vt_spont1.merge(Vt_spont2, on='charttime', how='outer')
        values = []
        
        # Calucalte average as needed otherwise just store available value
        for _, row in merged_TV_spont.iterrows():
            if not math.isnan(row['TV spont']) and not math.isnan(row['Spont Vt']):
                values.append((row['TV spont'] + row['Spont Vt'] / 2))
            elif not math.isnan(row['Spont Vt']):
                values.append(row['Spont Vt'])
            elif not math.isnan(row['TV spont']):
                values.append(row['TV spont'])
            else:
                values.append(np.nan)
        merged_TV_spont['value'] = values
        return merged_TV_spont[['charttime', 'value']]

    def _process_ECMO_present(self, data: pd.DataFrame) -> pd.DataFrame:
        """Function that determines if an ECMO device was present at the patients bed side"""
        
        
        data['value'] = data['ecmopatient']
        # Finding the last  value
        final_index = np.nan
        first_index = np.nan
        
        # Find first and last information on ECMO presence
        for index, row in data.iterrows():
            if not math.isnan(row['value']):
                if math.isnan(first_index):
                    first_index = index
                final_index = index
        
        # Create Dataframe that has information on ecmo presence between first and last mentioning of ecmo for every minute
        if not math.isnan(final_index) and not math.isnan(first_index):
            data_processed = pd.DataFrame(columns=['charttime'])
            data_processed['charttime'] = [charttime for charttime in np.arange(data['charttime'].at[first_index],
                                                                                data['charttime'].at[final_index], 1.0)]
            data_processed = data_processed.merge(data[['charttime', 'value']], on='charttime', how='left')
            last_value = np.nan
            values = []

            # Fill nan values with last known value
            for index, row in data_processed.iterrows():
                if not math.isnan(row['value']):
                    last_value = row['value']
                values.append(last_value)
            data_processed['value'] = last_value
        else:
            data_processed = pd.DataFrame(columns=['charttime', 'value'])

        return data_processed

    def _process_ECMO_FiO2(self, data: pd.DataFrame, present: pd.DataFrame) -> pd.DataFrame:
        """Function that imputes FiO2 value for ECMO usage if an ECMO device was present at the patients bed side"""
        
        # Initialuze Variabels
        data = data.rename(columns={'value': 'fio2'})
        data = present.merge(data[['charttime', 'fio2']], on='charttime', how='left')
        values = []

        # Store last know value for every timestamp where a ECMO was present, if no ECMO is present store NaN
        last_value = np.nan
        for index, row in data.iterrows():
            if row['value'] == 1:
                if not math.isnan(row['fio2']):
                    last_value = row['fio2']

            elif ['value'] == 0:
                last_value == np.nan
            values.append(last_value)
        data['value'] = values
        return data[['charttime', 'value']]

    def _process_ECMO_Blood_Flow(self, data: pd.DataFrame, present: pd.DataFrame) -> pd.DataFrame:
        """Function that imputes Blood Flow value for ECMO usage if an ECMO device was present at the patients bed side"""
        
        # Initialize Variabels
        data = data.rename(columns={'value': 'bloodflow'})
        data = present.merge(data[['charttime', 'bloodflow']], on='charttime', how='left')
        values = []
        last_value = np.nan

        # Store last know value for every timestamp where a ECMO was present, if no ECMO is present store NaN
        for index, row in data.iterrows():
            if row['value'] == 1:
                if not math.isnan(row['bloodflow']):
                    last_value = row['bloodflow']
            elif ['value'] == 0:
                last_value == np.nan
            values.append(last_value)
        data['value'] = values
        return data[['charttime', 'value']]

    def _process_ECMO_Gas_Flow(self, data: pd.DataFrame, present: pd.DataFrame) -> pd.DataFrame:
        """Function that imputes Gas Flow value for ECMO usage if an ECMO device was present at the patients bed side"""

        # Initialize Variabels
        data['sweepecmo'] = data['sweepecmo'].str.translate({ord(c): None for c in 'L'})
        data = present.merge(data[['charttime', 'sweepecmo']], on='charttime', how='left')
        data['sweepecmo'] = pd.to_numeric(data['sweepecmo'])
        values = []
        last_value = np.nan

        # Store last know value for every timestamp where a ECMO was present, if no ECMO is present store NaN
        for index, row in data.iterrows():
            if row['value'] == 1:
                if not math.isnan(row['sweepecmo']):
                    last_value = row['sweepecmo']

            elif ['value'] == 0:
                last_value == np.nan
            values.append(last_value)
        data['value'] = values
        return data[['charttime', 'value']]

    def _process_meds_converted(self, data: pd.DataFrame, full_times: pd.DataFrame) -> pd.DataFrame:
        """Function that converts IV drug data for one drug type stored with start and endtime to a dataframe where the drugrate is given for every minute"""
        
        # Initialize variables
        finished = pd.DataFrame(full_times, columns=['charttime'])
        times_input = []
        # Go through every instance of an iv administration
        for _, row in data.iterrows():
            time = []
            value = []
            df = pd.DataFrame(columns=['charttime', 'value'])
            
            # Fill dataframe for the whole time an iv runs with the corresponding drug rate
            for i in range(int(row['starttime']), int(row['endtime'])):
                time.append(i)
                value.append(row['rate'])
            df['charttime'] = time
            df['value'] = value
            times_input.append(df)
        
        # If no IV data is present add row with no input
        if len(data.index) == 0:
            df = pd.DataFrame(columns=['charttime', 'value'])
            df['charttime'] = full_times['charttime'].at[0]
            df['value'] = np.nan
            times_input.append(df)

        # Merge all occurences of the administration of the drug into one dataframe
        all_times = pd.concat(times_input)
        all_times = all_times.groupby('charttime').sum()
        finished = finished.merge(all_times, on='charttime', how='left')
        return finished

    """Processing functions for differetn drugs that insure only data that can be interpreted is used"""
    def _process_Furosemid(self, data: pd.DataFrame) -> pd.DataFrame:
        
        # Only use drug data that is stored in mg/hour
        filtered = data[data['rateuom'] == 'mg/hour']
        filtered = filtered.reset_index()
        del filtered['index']

        return filtered

    def _process_Norepinephrin(self, data: pd.DataFrame) -> pd.DataFrame:
        
        # Get all drug data that is stored in mg/kg/min and create new data frame with only relevant information
        filtered_mg = data[data['rateuom'] == 'mg/kg/min']
        filtered_mg_new = pd.DataFrame(columns=['starttime', 'endtime', 'rate1'])
        filtered_mg = filtered_mg.reset_index()
        del filtered_mg['index']

        # Get all drug data that is stored in mcg/kg/min and create new data frame with only relevant information
        filtered_mcg = data[data['rateuom'] == 'mcg/kg/min']
        filtered_mcg_new = pd.DataFrame(columns=['starttime', 'endtime', 'rate2'])
        filtered_mcg = filtered_mcg.reset_index()
        del filtered_mcg['index']

        # Convert mg to mcg by multiplying and storing result in new DF
        filtered_mg_new['rate1'] = filtered_mg['rate'] * 1000
        filtered_mg_new['starttime'] = filtered_mg['starttime']
        filtered_mg_new['endtime'] = filtered_mg['endtime']

        # Store data for mcg/kg/min in new dataframe
        filtered_mcg_new['rate2'] = filtered_mcg['rate']
        filtered_mcg_new['starttime'] = filtered_mcg['starttime']
        filtered_mcg_new['endtime'] = filtered_mcg['endtime']
        
        # Merge both dataframes
        merged = filtered_mcg_new.merge(filtered_mg_new, on=['starttime', 'endtime'], how='outer')
        values = []
        
         # Calucalte total as needed otherwise just store available value
        for _, row in merged.iterrows():
            if not math.isnan(row['rate1']) and not math.isnan(row['rate2']):
                values.append(row['rate1'] + row['rate2'])
            elif not math.isnan(row['rate1']):
                values.append(row['rate1'])
            elif not math.isnan(row['rate2']):
                values.append(row['rate2'])
            else:
                values.append(np.nan)
        
        # Create DF where only one rate exists
        merged['rate'] = values
        del merged['rate1']
        del merged['rate2']
        return merged

    def _process_Propofol(self, data: pd.DataFrame) -> pd.DataFrame:
        
        # Get all drug data that is stored in mg/kg/min and create new data frame with only relevant information
        filtered_kg = data[data['rateuom'] == 'mg/kg/min']
        filtered_kg = filtered_kg.reset_index()
        del filtered_kg['index']
        filtered_kg_new = pd.DataFrame(columns=['starttime', 'endtime', 'rate1'])
        
        # Get all drug data that is stored in mg/hour and create new data frame with only relevant information
        filtered = data[data['rateuom'] == 'mg/hour']
        filtered = filtered.reset_index()
        del filtered['index']

        # Fill data into new DF and convert  mg/kg/min to mg/hour
        filtered_kg_new['rate1'] = filtered_kg['rate'] * filtered_kg['patientweight'] * 60
        filtered_kg_new['starttime'] = filtered_kg['starttime']
        filtered_kg_new['endtime'] = filtered_kg['endtime']

        # Fill data into new DF
        filtered_new = pd.DataFrame(columns=['starttime', 'endtime', 'rate2'])
        filtered_new['rate2'] = filtered['rate']
        filtered_new['starttime'] = filtered['starttime']
        filtered_new['endtime'] = filtered['endtime']

        # Merge both information into one DF
        merged = filtered_new.merge(filtered_kg_new, on=['starttime', 'endtime'], how='outer')
        values = []

        # Calucalte total drugrate using all available rates
        for _, row in merged.iterrows():
            if not math.isnan(row['rate1']) and not math.isnan(row['rate2']):
                values.append(row['rate1'] + row['rate2'])
            elif not math.isnan(row['rate1']):
                values.append(row['rate1'])
            elif not math.isnan(row['rate2']):
                values.append(row['rate2'])
            else:
                values.append(np.nan)
        
        # Create DF using the calculated drug rates
        merged['rate'] = values
        del merged['rate1']
        del merged['rate2']

        return merged

    def _process_Vasopressin(self, data: pd.DataFrame) -> pd.DataFrame:
        
        # Get all drug data that is stored in units/min and create new data frame with only relevant information
        filtered_min = data[data['rateuom'] == 'units/min']
        filtered_min = filtered_min.reset_index()
        del filtered_min['index']
        filtered_min_new = pd.DataFrame(columns=['starttime', 'endtime', 'rate1'])

        # Get all drug data that is stored in units/hour and create new data frame with only relevant information
        filtered_h = data[data['rateuom'] == 'units/hour']
        filtered_h = filtered_h.reset_index()
        del filtered_h['index']
        filtered_h_new = pd.DataFrame(columns=['starttime', 'endtime', 'rate2'])


        # Fill data into new DF 
        filtered_min_new['rate1'] = filtered_min['rate']
        filtered_min_new['starttime'] = filtered_min['starttime']
        filtered_min_new['endtime'] = filtered_min['endtime']

        # Fill data into new DF and convert  units/hour to units/min
        filtered_h_new['rate2'] = filtered_h['rate'] / 60
        filtered_h_new['starttime'] = filtered_h['starttime']
        filtered_h_new['endtime'] = filtered_h['endtime']

        # Merge both information into one DF
        merged = filtered_h_new.merge(filtered_min_new, on=['starttime', 'endtime'], how='outer')
        values = []

        # Calculate total drugrate using all available rates
        for _, row in merged.iterrows():
            if not math.isnan(row['rate1']) and not math.isnan(row['rate2']):
                values.append(row['rate1'] + row['rate2'])
            elif not math.isnan(row['rate1']):
                values.append(row['rate1'])
            elif not math.isnan(row['rate2']):
                values.append(row['rate2'])
            else:
                values.append(np.nan)
        
        # Create DF using the calculated drug rates
        merged['rate'] = values
        del merged['rate1']
        del merged['rate2']

        return merged

    def _process_Fentanyl(self, data: pd.DataFrame) -> pd.DataFrame:
        
        # Get all drug data that is stored in mcg/hour and return it 
        filtered = data[data['rateuom'] == 'mcg/hour']
        filtered = filtered.reset_index()
        del filtered['index']
        return filtered

    def _process_Ketanest(self, data: pd.DataFrame) -> pd.DataFrame:
        
        # Get all drug data that is stored in mcg/kg/min and create new data frame with only relevant information
        filtered_mcg_kg = data[data['rateuom'] == 'mcg/kg/min']
        filtered_mcg_kg = filtered_mcg_kg.reset_index()
        del filtered_mcg_kg['index']
        filtered_mcg_kg_new = pd.DataFrame(columns=['starttime', 'endtime', 'rate1'])
        
        # Get all drug data that is stored in mcg/min and create new data frame with only relevant information
        filtered_mcg_min = data[data['rateuom'] == 'mcg/min']
        filtered_mcg_min = filtered_mcg_min.reset_index()
        del filtered_mcg_min['index']
        filtered_mcg_min_new = pd.DataFrame(columns=['starttime', 'endtime', 'rate2'])

        # Get all drug data that is stored in mg/hour and create new data frame with only relevant information
        filtered_mg_h = data[data['rateuom'] == 'mg/hour']
        filtered_mg_h = filtered_mg_h.reset_index()
        del filtered_mg_h['index']
        filtered_mg_h_new = pd.DataFrame(columns=['starttime', 'endtime', 'rate3'])

        # Get all drug data that is stored in mg/kg/hour and create new data frame with only relevant information
        filtered_mg_kg = data[data['rateuom'] == 'mg/kg/hour']
        filtered_mg_kg = filtered_mg_kg.reset_index()
        del filtered_mg_kg['index']
        filtered_mg_kg_new = pd.DataFrame(columns=['starttime', 'endtime', 'rate4'])

        # Fill data into new DF and convert  mcg/kg/min to mg/hour
        filtered_mcg_kg_new['rate1'] = filtered_mcg_kg['rate'] * 60 * filtered_mcg_kg['patientweight'] / 1000
        filtered_mcg_kg_new['starttime'] = filtered_mcg_kg['starttime']
        filtered_mcg_kg_new['endtime'] = filtered_mcg_kg['endtime']

        # Fill data into new DF and convert  mcg/min to mg/hour
        filtered_mcg_min_new['rate2'] = filtered_mcg_min['rate'] * 60 / 1000
        filtered_mcg_min_new['starttime'] = filtered_mcg_min['starttime']
        filtered_mcg_min_new['endtime'] = filtered_mcg_min['endtime']

        # Fill data into new DF
        filtered_mg_h_new['rate3'] = filtered_mg_h['rate']
        filtered_mg_h_new['starttime'] = filtered_mg_h['starttime']
        filtered_mg_h_new['endtime'] = filtered_mg_h['endtime']

        # Fill data into new DF and convert  mg/kg/hour to mg/hour
        filtered_mg_kg_new['rate4'] = filtered_mg_kg['rate'] * filtered_mg_kg['patientweight']
        filtered_mg_kg_new['starttime'] = filtered_mg_kg['starttime']
        filtered_mg_kg_new['endtime'] = filtered_mg_kg['endtime']

        # Merge all information into one DF
        data_frames = [filtered_mcg_kg_new, filtered_mcg_min_new, filtered_mg_h_new, filtered_mg_kg_new]
        merged = reduce(lambda left, right: pd.merge(left, right, on=['starttime', 'endtime'], how='outer'),
                        data_frames)
        values = []
        
        # Calculate total drugrate using all available rates
        for _, row in merged.iterrows():
            values.append(row[['rate1', 'rate2', 'rate3', 'rate4']].sum(min_count=1))

        # Create DF using the calculated drug rates
        merged['rate'] = values
        del merged['rate1']
        del merged['rate2']
        del merged['rate3']
        del merged['rate4']

        return merged

    def _process_Dexmedetomidin(self, data: pd.DataFrame) -> pd.DataFrame:
        
        # Get all drug data that is stored in mcg/kg/hour and create new data frame with only relevant information
        filtered_mcg_kg = data[data['rateuom'] == 'mcg/kg/hour']
        filtered_mcg_kg = filtered_mcg_kg.reset_index()
        del filtered_mcg_kg['index']
        filtered_mcg_kg_new = pd.DataFrame(columns=['starttime', 'endtime', 'rate1'])

        # Get all drug data that is stored in mcg/min and create new data frame with only relevant information
        filtered_mcg_min = data[data['rateuom'] == 'mcg/min']
        filtered_mcg_min = filtered_mcg_min.reset_index()
        del filtered_mcg_min['index']
        filtered_mcg_min_new = pd.DataFrame(columns=['starttime', 'endtime', 'rate2'])

        # Fill data into new DF and convert  mcg/kg/hour to mcg/kg/min
        filtered_mcg_kg_new['rate1'] = filtered_mcg_kg['rate'] / 60
        filtered_mcg_kg_new['starttime'] = filtered_mcg_kg['starttime']
        filtered_mcg_kg_new['endtime'] = filtered_mcg_kg['endtime']

        # Fill data into new DF and convert  mcg/min to mcg/kg/min
        filtered_mcg_min_new['rate2'] = filtered_mcg_min['rate'] / filtered_mcg_min['patientweight']
        filtered_mcg_min_new['starttime'] = filtered_mcg_min['starttime']
        filtered_mcg_min_new['endtime'] = filtered_mcg_min['endtime']

        # Merge both information into one DF
        merged = filtered_mcg_kg_new.merge(filtered_mcg_min_new, on=['starttime', 'endtime'], how='outer')
        values = []

        # Calculate total drugrate using all available rates
        for _, row in merged.iterrows():
            values.append(row[['rate1', 'rate2']].sum(min_count=1))
        
        # Create DF using the calculated drug rates
        merged['rate'] = values
        del merged['rate1']
        del merged['rate2']

        return merged

    def _process_Midazolam(self, data: pd.DataFrame) -> pd.DataFrame:
        
        # Get all drug data that is stored in mg/hour and return it 
        filtered = data[data['rateuom'] == 'mg/hour']
        filtered = filtered.reset_index()
        del filtered['index']
        return filtered

    def _process_Morphin(self, data: pd.DataFrame) -> pd.DataFrame:
        
        # Get all drug data that is stored in mg/hour and return it 
        filtered = data[data['rateuom'] == 'mg/hour']
        filtered = filtered.reset_index()
        del filtered['index']
        return filtered

    def _process_Milrinon(self, data: pd.DataFrame) -> pd.DataFrame:
        
        # Get all drug data that is stored in mcg/kg/min and return it 
        filtered = data[data['rateuom'] == 'mcg/kg/min']
        filtered = filtered.reset_index()
        del filtered['index']
        return filtered

    def _process_Dobutamin(self, data: pd.DataFrame) -> pd.DataFrame:
        
       # Get all drug data that is stored in mcg/kg/min and return it 
        filtered = data[data['rateuom'] == 'mcg/kg/min']
        filtered = filtered.reset_index()
        del filtered['index']
        return filtered

    def _process_Epinephrin(self, data: pd.DataFrame) -> pd.DataFrame:
        
        # Get all drug data that is stored in mcg/kg/min and return it 
        filtered = data[data['rateuom'] == 'mcg/kg/min']
        filtered = filtered.reset_index()
        del filtered['index']
        return filtered

    def _process_Rocuronium(self, data: pd.DataFrame) -> pd.DataFrame:
        
        # Get all drug data that is stored with no unit and return it 
        filtered = data[data['rateuom'] == '']
        filtered = filtered.reset_index()
        del filtered['index']
        return filtered
