
import math
import datetime as dt
from typing import Any




import numpy as np
from numpy import  ndarray
import pandas as pd

from extraction.IExtractor import IExtractor
from Lookup.uka_extraction_indicies import *

from Lookup.MIMICIV_item_dict import *
from Lookup.features_new import *

from filtering.IFilter import IFilter
from filtering.MIMICIVFilter import MIMICIVFilter
from preprocessing.MIMICIVPreprocessor import MIMICIVPreprocessor



"""Class that is used to extract data from the MIMICIV-database"""
class MIMICIVExtractor(IExtractor):
    def __init__(self, options: Any, cursor: Any, job_id: int):
        # Init class variables
        super().__init__(options, job_id)
        self.cursor = cursor
        self.preprocessor = MIMICIVPreprocessor(self.options, self.job_id)
        
        

    def extract_page(self, pagesize: int, offset: int) -> tuple[ndarray, ndarray]:
        """Function used to extract Df that contains all relevant information for the patient, the second return value contains 
        the total number of occurrences of the parameters. Each row in patient_data corresponds to one patient. Data is extracted for patients
        which are found in between rows offset and offset+pagesize of the patient table"""
        
        samples = IExtractor.generate_empty_page(pagesize, 66)
        
        
        # Get amount of admissions that are atleast 2h long with patients over 18 and an ICD-10 coded diagnosis
        admission_count = self._extract_admissions(samples, pagesize, offset)

        # Return empty df if no viable admissions are found
        print("Got Admissions")
        if admission_count == 0:
            print("\tDidn't find any admissions for ", (pagesize, offset))
            return self.preprocessor.generate_variable_page(), self.preprocessor.generate_variable_page()

        # Delete empty rows
        empty_row_count = pagesize - admission_count
        if empty_row_count > 0:
            samples = np.delete(samples, [t + admission_count for t in range(empty_row_count)], axis=1)
            
        admissions = samples.transpose()
        
        
        # Extract further data for admissions
        page, freq = self._extract_data(admissions)
        return page, freq


     
    def _extract_admissions(self, samples: ndarray,  pagesize: int, offset: int):
        """Function that extracts all admissions which have ICD-10 Codes, are at least 18 years old and where on intensive care unit 
        for atleast 120min and their ventilation times""" 
        

        # Extract all admissions satisfying the inclusion critiera    
        self.cursor.execute("""
            SELECT * FROM (
                SELECT
                p.subject_id, p.gender, i.hadm_id as admission_id, p.anchor_age as admit_age,
                SUM(TIMESTAMPDIFF(MINUTE, intime, outtime)) as admission_length
                FROM SMITH_MIMICIV.patients p 
                    INNER JOIN SMITH_MIMICIV.icustays i  ON p.subject_id = i.subject_id 
                    WHERE i.hadm_id IN (SELECT DISTINCT  di.hadm_id FROM SMITH_MIMICIV.diagnoses_icd di WHERE di.icd_version = 10 GROUP BY di.hadm_id )
                    GROUP BY i.hadm_id 
                    ORDER BY i.hadm_id
            ) AS d WHERE d.admit_age >=18 and admission_length >=120
            LIMIT %s offset %s;""", (pagesize, offset)
        )
        
        admissions = self.cursor.fetchall()
        if len(admissions) == 0:
            return 0
        print("After first extraction")

        # Get admission ids for further use
        admission_ids = []
        for admission in admissions:
            admission_ids.append(admission[2])
        

        # Get all admissions venitlator times and durations 
        ventilation_data, ventilation_data_implied, ventilated_ids = self._extract_ventilation_times(admission_ids)
        
        

        
        # Delete all non ventilated patients since we have no PEEP for them
        admissions = IFilter.filter_ids_extract_admissions(admissions, 2, ventilated_ids)
        
        
        # Make dict for position of admissions
        admission_ids_filtered = []
        for admission in admissions:
             admission_ids_filtered.append(admission[2])
        id_dict_filtered = {k: v for v,k in enumerate(admission_ids_filtered)}
        
        
        # Get List of ventilation data (start, end duration) for each admission
        mech_vent_duration, mech_vent_start, mech_vent_end = self._add_ventilation_data(ventilation_data, ventilation_data_implied, admission_ids_filtered, id_dict_filtered)
        print("After ventilation")
        
        
        
        # Get avg. Height of patients
        height = self._get_height(admission_ids_filtered, id_dict_filtered)
        
        # Get avg. weight of patients
        weight = self._get_weight(admission_ids_filtered, id_dict_filtered)
        

       

        
        # Extract ICD-10 coded admissions for every elgible patient from data base
        self.cursor.execute(f"""
            SELECT hadm_id, icd_code FROM diagnoses_icd  WHERE hadm_id IN ({str(admission_ids)[1:-1]}) AND icd_version = 10;
            """
        )

        diagnoses = self.cursor.fetchall()
        


        
        # Create List of patients with their core data: Subject Id, Gender, Admission, Length of ICU stay, Age, Diagnosis, Weight and Height
        for i in range(len(admissions)):
            

            
            # Match Icd10 codes to the patient
            adm_codes = [d[1] for d in diagnoses if d[0]== admissions[i][2]]
            adm_codes_str =  ','.join(str(item) for item in adm_codes)

            # Store core data 
            samples[SUBJECT_ID][i].append(admissions[i][0])
            samples[GENDER][i].append(admissions[i][1])
            samples[ADMISSION_ID][i].append(admissions[i][2])
            samples[AGE][i].append(admissions[i][3])
            samples[ADMISSION_LENGTH][i].append(admissions[i][4])
            samples[DIAGNOSIS][i] = adm_codes_str
            
            samples[HEIGHT][i] = height[i][0] if not math.isnan(height[i][0]) else np.nan
            samples[WEIGHT][i] = weight[i][0] if not math.isnan(weight[i][0]) else np.nan
            samples[MECHVENTSTART][i] = mech_vent_start[i]
            samples[MECHVENTEND][i] = mech_vent_end[i]
            samples[MECHVENTDURATION][i] = mech_vent_duration[i]
           
        
        return len(admissions)

    # Get Height of patients, due to multiple heights for each patient we take the average
    def _get_height(self, admission_ids: list, id_dict: dict) -> list:
        # Extract data from database
        self.cursor.execute(f"""
         SELECT hadm_id, valuenum FROM chartevents WHERE itemid = 226730 AND hadm_id in ({str(admission_ids)[1:-1]}) ORDER BY hadm_id;
         """)
        height_data = self.cursor.fetchall()
        
        height_list = [[] for _ in range(len(admission_ids))]
        height_list_avg = [[] for _ in range(len(admission_ids))]
        # Go through every height that is returned and store a list off all available heights for one admission in a list to calculate the average
        # if no height is recorded store nan.
        for i in range(len(height_data)):
            for j in range(len(admission_ids)):
                if admission_ids[j]==height_data[i][0]:
                    height_list[id_dict.get(admission_ids[j])].append(height_data[i][1])
        for i in range(len(height_list)):
            height_sum = 0
            for height in height_list[i]:
                height_sum += height
            if(len(height_list[i])) > 0:   
                height_list_avg[id_dict.get(admission_ids[i])].append(height_sum/len(height_list[i]))
            else: 
                height_list_avg[id_dict.get(admission_ids[i])].append(math.nan)
            
        return height_list_avg
    
     # Get Height of patients, due to multiple heights for each patient we take the average
    def _get_weight(self, admission_ids: list, id_dict: dict) -> list:
        
        # Get Patient weight in kilograms
        self.cursor.execute(f"""
         SELECT hadm_id, valuenum FROM chartevents WHERE itemid = 226512 AND hadm_id in ({str(admission_ids)[1:-1]}) ORDER BY hadm_id;
         """)
        weight_data = self.cursor.fetchall()

        # Get patient weight in pounds
        self.cursor.execute(f"""
         SELECT hadm_id, valuenum FROM chartevents WHERE itemid = 226531 AND hadm_id in ({str(admission_ids)[1:-1]}) ORDER BY hadm_id;
         """)
        weight_data_lbs = self.cursor.fetchall()
        
        weight_list = [[] for _ in range(len(admission_ids))]
        weight_list_avg = [[] for _ in range(len(admission_ids))]
        for j in range(len(admission_ids)):
            # Add weights to weight list of each patient
            for i in range(len(weight_data)):
                if admission_ids[j]==weight_data[i][0]:
                    weight_list[id_dict.get(admission_ids[j])].append(weight_data[i][1])
            for h in range(len(weight_data_lbs)):
                if admission_ids[j] == weight_data_lbs[h]:
                    # Convert pound to kg
                    weight_kg = weight_data_lbs[h][1]*0.45359237

                    weight_list[id_dict.get(admission_ids[j])].append(weight_kg)
        for i in range(len(weight_list)):
            # Store average weight of patient as patient weight
            weight_sum = 0
            for weight in weight_list[i]:
                weight_sum += weight
            if(len(weight_list[i])) > 0:   
                weight_list_avg[id_dict.get(admission_ids[i])].append(weight_sum/len(weight_list[i]))
            else: 
                weight_list_avg[id_dict.get(admission_ids[i])].append(math.nan)

        
        return weight_list_avg


    # Get Start, end and duration of ventilation
    def _extract_ventilation_times(self,  admission_ids: list):
        """Functions that extract all rows that contain information on ventilation"""
        
        # Query for extraction
        query = f"""
            select p.hadm_id, timestampdiff(minute, starttime , endtime) as ventilation_length, p.starttime, p.endtime from SMITH_MIMICIV.procedureevents p
               
            where itemid in (225792, 225794) AND hadm_id IN ({str(admission_ids)[1:-1]});
        """
        
        self.cursor.execute(query)
        # Stores end admission_id, start and end time as well as duration of invasive and non invasive ventilation
        ventilation_data = self.cursor.fetchall()

        # Get values that imply ventilation
        query = f"""
            SELECT hadm_id, charttime, itemid FROM SMITH_MIMICIV.chartevents where itemid in (224695, 224696, 220339, 220339, 226873) and hadm_id IN ({str(admission_ids)[1:-1]});
        """
        self.cursor.execute(query)
        
        # Stores values that imply ventilation
        ventilation_data_implied =  pd.DataFrame(self.cursor.fetchall(), columns=self.cursor.column_names)
        

        # Contains hadm_id of each ventilated patient
        ventilated_ids_list = []
        ventilated_ids = set()

        # Get ids of all ventilated patients
        for ventilation in ventilation_data:
            ventilated_ids_list.append(ventilation[0])
        for entry in ventilation_data_implied:
            ventilated_ids_list.append(entry[0])
        ventilated_ids.update(ventilated_ids_list)
        print("Got vent data")
        return ventilation_data, ventilation_data_implied, ventilated_ids

    
    def _add_ventilation_data(self, ventilation_data: ndarray, ventilation_data_implied: pd.DataFrame, ventilation_ids, id_dict: dict) -> list:
        """Function that determines start, end and duration of the ventilation of each patient. One patient may have multiple periods of ventilation."""

        # Contains start times for each ventilated patient, each patient has its on list to accomodate multiple ventilations
        start_time = [[] for _ in range(len(ventilation_ids))]
        # Contains end times for each ventilated patient, each patient has its on list to accomodate multiple ventilations
        end_time = [[] for _ in range(len(ventilation_ids))]
        # Contains duration of ventilations for each ventilated patient, each patient has its on list to accomodate multiple ventilations
        duration = [[] for _ in range(len(ventilation_ids))]
        
        
        # Iterate over ventilation data and fill lists with start and end time, duration of each ventilation for each patient
        for j in range(len(ventilation_ids)):
            admission_id = ventilation_ids[j]
            # Add all recorded times for mechanical ventilation
            for i in range(len(ventilation_data)):
                if admission_id == ventilation_data[i][0]:
                    duration[id_dict.get(admission_id)].append(ventilation_data[i][1])
                    start_time[id_dict.get(admission_id)].append(ventilation_data[i][2])
                    end_time[id_dict.get(admission_id)].append(ventilation_data[i][3])
            
            # Get data for implied mechanical ventilation by appearance of values for PEEP, P_EI or I:E ratio
            ventilation_admission = ventilation_data_implied[ventilation_data_implied['hadm_id'] == admission_id]
            ventilation_admission = ventilation_admission.sort_values(by='charttime')
            if not len(ventilation_admission.index) == 0 :
                ventilation_admission.reset_index(inplace=True)
                del ventilation_admission['index']
                start_time_admission = start_time[id_dict.get(admission_id)]
                end_time_admission = end_time[id_dict.get(admission_id)]
                implied_times = []
                
                # Check if implied ventilation is covered by known charted ventilation
                for h in range(len(ventilation_admission.index)):

                    implied_covered = False
                    implied_time = ventilation_admission['charttime'].at[h]
                    for i in range(len(start_time_admission)) :
                        start = start_time_admission[i]
                        end = end_time_admission[i]
                        if  start <= implied_time <= end:
                            implied_covered = True
                            break
                    if not implied_covered:
                        implied_times.append(implied_time)
                
                # Add implied ventilation not covered by recorded ventilation
                ventilation_tuple = self.preprocessor.process_implied_Ventilation(implied_times)
                duration[id_dict.get(admission_id)] = duration[id_dict.get(admission_id)] + ventilation_tuple[2]
                start_time[id_dict.get(admission_id)] = start_time[id_dict.get(admission_id)] +ventilation_tuple[0]
                end_time[id_dict.get(admission_id)] = end_time[id_dict.get(admission_id)] + ventilation_tuple[1]



                        
        return duration, start_time, end_time

    def _extract_data(self, admissions: ndarray):
        """Function that extracts all relevant parameters for the patients that are contained in samples. It returns a dataframe with the features used 
        during the learning process as well as the total number of occurrences of the parameters"""
        
        print("Data extraction")
        # Determines how many horowitzvalues  are averaged
        windowsize_horowitz = self.options["extraction_parameter"]["windowsize_horowitz"]
        
        # Determines max time in between horowitz values before a new series gets started
        cutoff_horowitz_time = self.options["extraction_parameter"]["cutoff_time_horowitz"]
        
        #Dertermines timespan in minutes (before and after) that are taken into account from point of lowest horowitz
        #Note the -15 in windowsize_data after are caused by the fact that the timestamp of min horowitz already count as after occurence of ARDS
        days_before = self.options["extraction_parameter"]["windowsize_before"]
        days_after = self.options["extraction_parameter"]["windowsize_after"]
        windowsize_data_before = days_before * 24 * 60
        windowsize_data_after = days_after * 24 * 60 

        # Initialize variables
        admission_ids = []
        list_values_fiO2 = []
        list_abs_times_fiO2 = []
        list_times_fiO2 = []
        list_values_paO2 = []
        list_abs_times_paO2 = []
        list_times_paO2 = []
        list_values_PEEP = []
        list_abs_times_PEEP = []
        list_times_PEEP = []
        id_dict = {}
        
        # Ensure one entry in each lst for each admission and build dictionary
        for index, admission in enumerate(admissions):
                admission_id = int(str(admission[ADMISSION_ID])[1:-1])

                admission_ids.append(admission_id)
                list_abs_times_PEEP.append([])
                list_abs_times_fiO2.append([])
                list_abs_times_paO2.append([])
                list_times_PEEP.append([])
                list_times_fiO2.append([])
                list_times_paO2.append([])
                list_values_fiO2.append([])
                list_values_paO2.append([])
                list_values_PEEP.append([])
                id_dict[admission_id] = index
        
        #Get data for min Horowitz values
        query = f"SELECT cn.hadm_id, itemid, timestampdiff(minute, a.admittime, cn.charttime), valuenum, cn.charttime FROM chartevents cn INNER JOIN admissions a ON  a.hadm_id = cn.hadm_id WHERE cn.hadm_id IN ({str(admission_ids)[1:-1]}) AND (itemid = 223835 OR itemid = 220224 OR itemid = 220339);"
        
        self.cursor.execute(query)
        data = self.cursor.fetchall()
        print("data Horowitz fetched")
        
        
        # Add data to corresponding lists
        
        for row in data:
            admission_id = row[0]
            
            index = id_dict.get(admission_id)
            #list_times_all[index].append(row[0])
            if row[1] == 223835:
                list_values_fiO2[index].append(row[3]/100)
                list_abs_times_fiO2[index].append(row[2])
                list_times_fiO2[index].append(row[4])
            elif row[1] == 220224:
                list_abs_times_paO2[index].append(row[2])
                list_values_paO2[index].append(row[3])
                list_times_paO2[index].append(row[4])
            elif row[1] == 220339:
                    list_values_PEEP[index].append(row[3])
                    list_abs_times_PEEP[index].append(row[2])
                    list_times_PEEP[index].append(row[4])
            else:
                print("Unexpected item id Horowitz extraction! itemid: " + str(row[1]))
        
        # Calculate horovitz-indices and time of minimal horovitz time
        patients_horowitz_df, min_horowitz_times = self.preprocessor._calculate_horowitz(admissions, id_dict, list_abs_times_PEEP, list_times_PEEP, list_values_PEEP,
                                                    list_abs_times_fiO2, list_values_fiO2, list_times_fiO2, list_abs_times_paO2, list_values_paO2, list_times_paO2)
        print("Data horowitz")
        
        
        

        # Filter out patients that have no horowitz value
        
        admissions = IFilter.filter_ids_extract_data(admissions, 1, list(patients_horowitz_df['admission_id']))
        if len(admissions) == 0:
            return self.preprocessor.generate_feature_page(), self.preprocessor.generate_variable_page()
           

        admission_ids = []
        height_dict = {}
        gender_dict = {}
        ventilation_dict = {}
        
        # Create dict for mapping patient id to their weight and patient id to their index in admissions and store information regarding ventilation
        for index, admission in enumerate(admissions):
                admission_id = int(str(admission[ADMISSION_ID])[1:-1])
                admission_height = admission[HEIGHT]
                admission_gender = str(admission[GENDER])[1:-1]
                admission_vent_start = admission[MECHVENTSTART]
                admission_vent_end =  admission[MECHVENTEND]

                admission_vent_info = pd.DataFrame(columns=['start', 'end'])
                admission_vent_info['start'] = admission_vent_start
                admission_vent_info['end'] = admission_vent_end

                admission_ids.append(admission_id)
                
                
                gender_dict[admission_id] = admission_gender[1:-1]
                height_dict[admission_id] = admission_height
                
                ventilation_dict[admission_id] = admission_vent_info
                
        # Get start and end of admission
        query = f"SELECT hadm_id, admittime, dischtime FROM admissions WHERE hadm_id in ({str(admission_ids)[1:-1]}) ;"
        self.cursor.execute(query)
        admission_start = pd.DataFrame(self.cursor.fetchall())
        admission_start.columns = ['admission_id', 'admittime', 'dischtime']
        admission_times = admission_start.merge(min_horowitz_times, how='left', on='admission_id')
        

        # Get output volume
        query_vol_output = f"""SELECT hadm_id as admission_id, value as value, charttime  
                FROM outputevents   WHERE  valueuom = 'ml' and hadm_id IN({str(admission_ids)[1:-1]}) ORDER BY charttime ASC; """
        
        self.cursor.execute(query_vol_output)
        data_vol_output = pd.DataFrame(self.cursor.fetchall(), columns=self.cursor.column_names)
        
        
        #Get input volume
        query_vol_input = f"""SELECT i.hadm_id as admission_id, i.amount as value, i.starttime as starttime, i.endtime as endtime, ordercategorydescription as way
                FROM inputevents i   WHERE  amountuom = 'ml' and i.hadm_id IN({str(admission_ids)[1:-1]}) ORDER BY starttime ASC; """
        self.cursor.execute(query_vol_input)
        data_vol_input = pd.DataFrame(self.cursor.fetchall(), columns=self.cursor.column_names)
        
        

        # Get relevant inputevents
        query_input = f"""SELECT hadm_id as admission_id, itemid, starttime, endtime, amount as value, rate, amountuom, rateuom, ordercategorydescription as way, patientweight FROM inputevents
                        WHERE hadm_id IN({str(admission_ids)[1:-1]}) 
                        AND itemid in ({str(list(input_item_dict.values()))[1:-1]}) ORDER by starttime ASC;"""
        
        self.cursor.execute(query_input)
        data_input = pd.DataFrame(self.cursor.fetchall(), columns= self.cursor.column_names)
        
        

        # Get relevant chartevents
        query_chart = f"""SELECT hadm_id as admission_id, itemid, charttime, valuenum as value,
                        CASE WHEN itemid = 224093 AND  value = 'Prone' THEN 1
                            WHEN itemid = 224093 AND NOT value = 'Prone' THEN 0 
                            ELSE NULL 
                        END AS patientpos,
                        (CASE WHEN itemid = 229267 AND value = 'YES' THEN 1
                            WHEN itemid = 229267 AND NOT value = 'YES' THEN 0
                            ELSE NULL
                        END) AS ecmopatient,
                        (CASE WHEN itemid = 229278 THEN value
                            ELSE NULL
                        END) AS sweepecmo
                        FROM chartevents 
                        WHERE hadm_id IN({str(admission_ids)[1:-1]})
                        AND itemid in ({str(list(chart_item_dict.values()))[1:-1]}) ORDER BY charttime ASC;"""
        self.cursor.execute(query_chart)
        data_chart = pd.DataFrame(self.cursor.fetchall(), columns= self.cursor.column_names)
        
        
        
        # Get relevant labevents
        query_lab = f"""SELECT hadm_id as admission_id, itemid, charttime, valuenum as value FROM labevents
                        WHERE hadm_id IN({str(admission_ids)[1:-1]}) 
                        AND itemid in ({str(list(lab_item_dict.values()))[1:-1]}) ORDER BY charttime ASC;"""
        self.cursor.execute(query_lab)
        
        data_lab = pd.DataFrame(self.cursor.fetchall(), columns= self.cursor.column_names)
        

        # Get procedures
        query_procedures = f"""SELECT hadm_id as admission_id, itemid, starttime, endtime, value, valueuom FROM procedureevents 
                        WHERE itemid in ({str(list(procedure_item_dict.values()))[1:-1]})
                        AND hadm_id in ({str(admission_ids)[1:-1]}) ORDER BY starttime ASC"""
        self.cursor.execute(query_procedures)
        data_procedures = pd.DataFrame(self.cursor.fetchall(), columns= self.cursor.column_names)
        
        print("Data fechted")
        

        
        
        patient_list = []
        patient_list_freq = []

       
        for admission in admissions :
            admission_id = admission[ADMISSION_ID][0]
            
            # Get relevant data
            patient_admission_times = admission_times.query('admission_id == @admission_id')
            patient_admission_times.reset_index(inplace=True)
            admit_time = patient_admission_times['admittime'].at[0]
            min_horowitz_timestamp = patient_admission_times['charttime'].at[0]
            min_horowitz_times_minutes = patient_admission_times['abstime'].at[0]

            # Calculate begin and end of window
            time_data_start = min_horowitz_timestamp - dt.timedelta(minutes=windowsize_data_before)
            time_data_end = min_horowitz_timestamp + dt.timedelta(minutes=(windowsize_data_after-1))

            time_window_end = (time_data_end-admit_time).total_seconds()/60
            time_window_start = (time_data_start-admit_time).total_seconds()/60

            # Get data for specific patient out of all data
            patient_vol_input = data_vol_input.query('admission_id == @admission_id')
            patient_vol_input = patient_vol_input.reset_index()
            del patient_vol_input['index']

            patient_vol_output = data_vol_output.query('admission_id == @admission_id')
            patient_vol_output = patient_vol_output.reset_index()
            del patient_vol_output['index']
            

            patient_input = data_input.query('admission_id == @admission_id')
            patient_input = patient_input.reset_index()
            del patient_input['index']

            patient_lab = data_lab.query('admission_id == @admission_id')
            patient_lab = patient_lab.reset_index()
            del patient_lab['index']

            patient_chart = data_chart.query('admission_id == @admission_id')
            patient_chart = patient_chart.reset_index()
            del patient_chart['index']
            
            patient_procedure = data_procedures.query('admission_id == @admission_id')
            patient_procedure = patient_procedure.reset_index()
            del patient_procedure['index']

            patient_horowitz = patients_horowitz_df.query('admission_id == @admission_id')
            patient_horowitz = patient_horowitz.reset_index()
            del patient_horowitz['index']

            filter_MIMICIV = MIMICIVFilter(self.options)
            
            # Delete all data not in window and convert timestamps to minutes since admission
            patient_vol_input = filter_MIMICIV._delete_data_not_in_window_input(patient_vol_input, time_data_start, time_data_end, admit_time)
            patient_input = filter_MIMICIV._delete_data_not_in_window_input(patient_input, time_data_start, time_data_end, admit_time)

            patient_procedure = filter_MIMICIV._delete_data_not_in_window_procedure(patient_procedure, time_data_start, time_data_end, admit_time)
            
            patient_vol_output = filter_MIMICIV._delete_data_not_in_window(patient_vol_output, time_data_start, time_data_end, admit_time)
            patient_chart = filter_MIMICIV._delete_data_not_in_window(patient_chart, time_data_start, time_data_end, admit_time)
            

            patient_lab = filter_MIMICIV._delete_data_not_in_window(patient_lab, time_data_start, time_data_end, admit_time)
            patient_horowitz = filter_MIMICIV._delete_data_not_in_window(patient_horowitz, time_data_start, time_data_end, admit_time)

            patient_ventilation = filter_MIMICIV._delete_data_not_in_window_ventilation(ventilation_dict.get(admission_id), time_data_start, time_data_end, admit_time)

            #Create Lists and Dataframes for different data 
            list_chart_df = [[] for _ in range(len(chart_index_dict))]
            list_lab_df = [[] for _ in range(len(lab_index_dict))]
            list_procedures_df = [[] for _ in range(len(procedure_index_dict))]
            list_input_df = [[] for _ in range(len(input_index_dict))]
            
            # Fill list with their corresponding data and create Dataframes
            for itemid in chart_item_dict.values():
                
                data = patient_chart.query('itemid == @itemid')
                data = data.reset_index()
                del data['index']
                data = data.infer_objects()
                if itemid == 229267:
                    # Get information on ECMO usage
                    data  = self.preprocessor._process_ECMO_present(data).astype('float64')
                    
                list_chart_df[chart_index_dict.get(itemid)] = data 
            
            
            for itemid in lab_item_dict.values():
                
                data = patient_lab.query('itemid == @itemid')
                data = data.reset_index()
                del data['index']
                
                list_lab_df[lab_index_dict.get(itemid)] = data.infer_objects()
            
           
            
            for itemid in input_item_dict.values():
                
                data = patient_input.query('itemid == @itemid')
                data = data.reset_index()
                del data['index']
                
                
                list_input_df[input_index_dict.get(itemid)] = data.infer_objects()
            
            for itemid in procedure_item_dict.values():
                data = patient_procedure.query('itemid == @itemid')
                data = data.reset_index()
                del data['index']
                 
                list_procedures_df[procedure_index_dict.get(itemid)] = data.infer_objects()

            # Create timetable for whole window
            times = [float(i) for i in np.arange(time_window_start, (time_window_end+1), 1.0)]

            print("Admission_ID: " + str(admission_id))

            # Process extracted data
            data, freq = self.preprocessor.process_data_new(list_lab_df, list_chart_df, list_input_df, list_procedures_df, patient_ventilation,  patient_horowitz, patient_vol_input, patient_vol_output, admission, times, min_horowitz_times_minutes)
            patient_list_freq.append(freq)
            patient_list.append(data)
            

            

        return pd.concat(patient_list),  pd.concat(patient_list_freq)

    
    
    

    
    

