
from datetime import timedelta
import string
from typing import Any
from functools import reduce
import re


import numpy as np
from numpy import ndarray
import pandas as pd


from extraction.IExtractor import IExtractor
from preprocessing.eICUPreprocessor import eICUPreprocessor
from filtering.eICUFilter import eICUFilter
from Lookup.uka_extraction_indicies import *
from Lookup.diagnosis_indicies import diagnosis_dict
from Lookup.Feature_mapping_eICU import *

"""Class that is used to extract data from the eICU-database"""
class eICUExtractor(IExtractor):
    def __init__(self, options: Any, cursor: Any, job_id:int):
        super().__init__(options, job_id)
        self.cursor = cursor
        self.dict_stays_times = {}
        unit_stays_df = pd.DataFrame(columns=["patienthealthsystemstayid", "unitid", "unitadmittime24", "unitdischargetime24", "hospitaladmitoffset", "unitdischargeoffset"])
        self.preprocessor = eICUPreprocessor(self.options, unit_stays_df, self.job_id)
        
    
    def extract_page(self, pagesize: int, offset: int) -> tuple[ndarray, ndarray]:
        """Function used to extract Df that contains all relevant information for the patient, the second return value contains 
        the total number of occurrences of the parameters. Each row in patient_data corresponds to one patient. Data is extracted for patients
        which are found in between rows offset and offset+pagesize of the patient table"""
        
        samples = IExtractor.generate_empty_page(pagesize, 12)
        
        # Get amount of admissions that are atleast 2h long with patients over 18 and an ICD-10 coded diagnosis
        admission_count = self._extract_admissions(samples, pagesize, offset)

        # Return empty df if no viable admissions are found
        if admission_count == 0:
            print("\tDidn't find any admissions for ", (pagesize, offset))
            return self.preprocessor.generate_feature_page(), self.preprocessor.generate_variable_page()

        # Delete empty rows
        empty_row_count = pagesize - admission_count
        if empty_row_count > 0:
            samples = np.delete(samples, [t + admission_count for t in range(empty_row_count)], axis=1)
        admissions = samples.transpose()
        
        # Extract further data for admissions and get final feature vector along with frequiences of the parameters
        page, freq = self._extract_data(admissions)
        return page, freq
        
    
    def _extract_admissions(self, samples: ndarray, pagesize: int, offset: int):
        """Function that extracts all admissions which have ICD-10 Codes, are at least 18 years old and where on intensive care unit 
        for atleast 120min and their ventilation times""" 
        

        # Extract all admissions satisfying the inclusion critiera 
        self.cursor.execute(
            "SELECT * FROM ("
            "   SELECT"
            "       p.uniquepid as upid , p.gender as gender , p.patienthealthsystemstayid as hadm_id,"
            "       (CASE WHEN age = '> 89' THEN 90"
            "           ELSE p.age END) as age,"
            "       p.admissionheight as height, p.admissionweight as weight,"
            "       SUM(unitdischargeoffset) as admission_length, p.hospitaladmittime24 as h_admit_time FROM patient p "
            "       GROUP BY p.patienthealthsystemstayid"
            "       ORDER BY p.patienthealthsystemstayid"
            ") AS d WHERE (d.age = '> 89' or d.age>=18) AND d.admission_length>= 120"
            "   LIMIT %s OFFSET %s", (pagesize, offset)
        )

        # Convert to correct datatype
        admissions = pd.DataFrame(self.cursor.fetchall(), columns=self.cursor.column_names)
        admissions['hadm_id'] = pd.to_numeric(admissions['hadm_id'])
        admissions['admission_length'] = pd.to_numeric(admissions['admission_length'])
        admissions['h_admit_time'] = pd.to_datetime(admissions['h_admit_time'])
        admissions['age'] = pd.to_numeric(admissions['age'])
        admission_ids = []
        ids_non_icd10 = []
        for index, row in admissions.iterrows():
            admission_ids.append(row['hadm_id'])
            ids_non_icd10.append(row['hadm_id'])
        
        # Extract all diagnosis
        self.cursor.execute(f"""
            SELECT c.patienthealthsystemstayid, d.icd9code  FROM diagnosis d 
                INNER JOIN (SELECT p.patientunitstayid, p.patienthealthsystemstayid FROM patient p 
                WHERE p.patienthealthsystemstayid in ({str(admission_ids)[1:-1]}) GROUP BY patientunitstayid) as c 
                ON c.patientunitstayid = d.patientunitstayid 
                ORDER BY c.patienthealthsystemstayid;"""
        )
        diagnosis = self.cursor.fetchall()
        
        # Remove non icd 10 code admissions
        for i in range(len(diagnosis)):
            if re.search('[a-zA-Z]', diagnosis[i][1]):
                try: 
                    ids_non_icd10.remove(int(diagnosis[i][0]))
                except:
                    pass
        admissions = admissions.query(' hadm_id not in @ids_non_icd10')
        admissions = admissions.reset_index()
        del admissions['index']
        admission_ids = [x for x in admission_ids if x not in ids_non_icd10]

        # Remove all diagnosis where the patient has not at least one ICD-10 diagnosis
        diagnosis = [d for d in diagnosis if not d[0] in ids_non_icd10]
        
        
        # Get the Unit stay ids for further use as primary keys
        self.cursor.execute(f"""
            SELECT patienthealthsystemstayid, patientunitstayid as unitid, unitadmittime24, unitdischargetime24, hospitaladmitoffset, unitdischargeoffset  FROM patient WHERE patienthealthsystemstayid IN ({str(admission_ids)[1:-1]});
        """)
        stay_ids = pd.DataFrame(self.cursor.fetchall(), columns=self.cursor.column_names)
        stay_ids['unitadmittime24'] = pd.to_datetime(stay_ids["unitadmittime24"])
        stay_ids['unitdischargetime24'] = pd.to_datetime(stay_ids["unitdischargetime24"])
        stay_ids['unitdischargeoffset'] = pd.to_numeric(stay_ids["unitdischargeoffset"])
        stay_ids['unitid'] = pd.to_numeric(stay_ids["unitid"])
        stay_ids['hospitaladmitoffset'] = pd.to_numeric(stay_ids["hospitaladmitoffset"])
        stay_ids['patienthealthsystemstayid'] = pd.to_numeric(stay_ids['patienthealthsystemstayid'])
        
        stay_ids = stay_ids.sort_values(by='unitadmittime24', ascending=True)
        stay_ids = stay_ids.reset_index()
        del stay_ids['index']
        
        unit_stays_list = []
        unit_stay_ids = []

        # Get begin, end and id of each unit stay for each admission
        for index, adm_row in admissions.iterrows():
            adm_id = adm_row["hadm_id"]
            
            # Get relevant unitstays for this admission
            unitstays = stay_ids[(stay_ids['patienthealthsystemstayid'])== adm_id]
            
            admit_times = []
            discharge_times = []

            # Create list of all admission and discharge times during one admission and store them
            for i, stayrow in unitstays.iterrows():
                unit_adm_time = abs(unitstays.loc[i, 'hospitaladmitoffset'])
                unit_discharge = stayrow['unitdischargeoffset']
                admit_times.append(unit_adm_time)
                discharge_times.append(unit_adm_time + unit_discharge)
            unitstays['unitadmittime24'] = admit_times
            unitstays['unitdischargetime24'] = discharge_times
            unit_stays_list.append(unitstays)
            
            # Create list of unit stay ids of each relevant admission
            unit_stay_ids = unit_stay_ids + unitstays['unitid'].values.tolist()
        
        # Get information for parameters that indicate ventilation 
        self.cursor.execute(f"""
                 SELECT patientunitstayid as unitid, respchartoffset as charttime FROM SMITH_eICU.respiratorycharting r  
                 WHERE  (respchartvaluelabel = 'PEEP' or respchartvaluelabel = 'PEEP/CPAP'  or respchartvaluelabel = 'Plateau Pressure' or respchartvaluelabel = 'Peak Insp. Pressure'  or respchartvaluelabel = 'PD I:E RATIO') AND patientunitstayid in ({str(unit_stay_ids)[1:-1]});
            """)
        data_ventilation = pd.DataFrame(self.cursor.fetchall(), columns=self.cursor.column_names)
        
        # Convert to correct data type
        data_ventilation['unitid'] = pd.to_numeric(data_ventilation["unitid"])
        data_ventilation['charttime'] = pd.to_numeric(data_ventilation["charttime"])
        
        # Process ventilation information
        unit_stays_df = pd.concat(unit_stays_list)
        self.preprocessor.update_unitstays(unit_stays_df)
        data_ventilation = self.preprocessor.create_time(data_ventilation)
        vent_info_dict = self.preprocessor.predict_ventilation(data_ventilation, admission_ids)
        
        # Store extracted information for each admission in an array
        for index, adm_row in admissions.iterrows():

            # Get relevant diagnosis for admission
            adm_codes = [d[1] for d in diagnosis if d[0] == str(adm_row["hadm_id"])]
            str_diagnosis = ','.join(adm_codes)

            # Store basic information
            adm_id = adm_row["hadm_id"]
            samples[SUBJECT_ID][index].append(adm_row['upid'])
            samples[UNIT_STAY_INFO][index] = unit_stays_list[index]
            samples[GENDER][index].append(adm_row['gender'])
            samples[ADMISSION_ID][index].append(adm_id)
            samples[AGE][index].append(adm_row['age'])
            samples[ADMISSION_LENGTH][index].append(adm_row['admission_length'])
            samples[DIAGNOSIS][index] = str_diagnosis
            samples[HEIGHT][index] = float(adm_row['height']) if not adm_row['height'] == '' else np.nan
            samples[WEIGHT][index] = float(adm_row['weight']) if not adm_row['weight'] == '' else np.nan
            samples[MECHVENTSTART][index] = vent_info_dict.get(adm_id)[0]
            samples[MECHVENTEND][index] = vent_info_dict.get(adm_id)[1]
            samples[MECHVENTDURATION][index] = vent_info_dict.get(adm_id)[2]
        return (len(admissions))


   

    def _extract_data(self, admissions: ndarray):
        """Function that extracts all relevant parameters for the patients that are contained in samples. It returns a dataframe with the features used 
        during the learning process as well as the total number of occurrences of the parameters"""

        # Get size of window and transform duration into minutes
        days_before = self.options["extraction_parameter"]["windowsize_before"]
        days_after = self.options["extraction_parameter"]["windowsize_after"]
        windowsize_data_before = days_before * 24 * 60
        windowsize_data_after = days_after * 24 * 60

        # Initializes variables
        window_data = pd.DataFrame(columns=['hadm_id', 'starttime', 'endtime'])
        relevant_stays_list = []
        list_adm_ids = []
        list_stays = []
        dict_adm_stays = {}
        dict_stays_adm = {}
        list_unit_stays = []
        dict_stay_index = {}
        dict_unit_admission = {}
        dict_unit_discharge = {}
        dict_adm_times = {}

        # Create dictionaries to map between admissions and unitstays
        for admission in admissions:
           
            admission_id = int(str(admission[ADMISSION_ID])[1:-1])
            list_adm_ids.append(admission_id)
            
            # Create dict to map between admission id and unit stay id
            dict_adm_stays[admission_id] = admission[UNIT_STAY_INFO]['unitid'].to_list()
            list_unit_stays.append(admission[UNIT_STAY_INFO])
            
            # Create dictionaries to mapt between unit stay id and admission id, index in the unit stay info df, as well as unit
            # admission and discharge times
            for index, row  in admission[UNIT_STAY_INFO].iterrows():
                list_stays.append(row['unitid'])
                dict_stays_adm[row['unitid']] = admission_id
                dict_stay_index[row['unitid']] = index
                dict_unit_admission[row['unitid']] = row['unitadmittime24']
                dict_unit_discharge[row['unitid']] = row['unitdischargetime24']
        
        # Update relevant unit stays id in the preprocessor
        unit_stays_df = pd.concat(list_unit_stays)
        self.preprocessor.update_unitstays(unit_stays_df)
        
    
        # Data Extraction

        # Extract data needed for horovitz-index calculation
        self.cursor.execute(f"""
             SELECT patientunitstayid as unitid, respchartoffset as charttime, respchartvaluelabel  as name, respchartvalue as value FROM SMITH_eICU.respiratorycharting r  
             WHERE  respchartvaluelabel in ('PEEP', 'FiO2') AND patientunitstayid in ({str(list_stays)[1:-1]});
        """)
        data_respiratorycharting_df = pd.DataFrame(self.cursor.fetchall(), columns=self.cursor.column_names)
        self.cursor.execute(f"""
            SELECT patientunitstayid as unitid, labresultoffset as charttime, labname as name,  labresult as value FROM lab WHERE (labname = 'paO2' or labname = 'FiO2' or labname = 'PEEP') AND patientunitstayid in ({str(list_stays)[1:-1]});
        """)
        data_lab_df = pd.DataFrame(self.cursor.fetchall(), columns=self.cursor.column_names)
        
        
        # Convert rows that contain numbers to numeric since all columns are string in the DB
        data_lab_df['value'] = data_lab_df['value'].str.replace(r'\%', '')
        data_respiratorycharting_df['unitid'] = pd.to_numeric(data_respiratorycharting_df["unitid"])
        data_respiratorycharting_df['charttime'] = pd.to_numeric(data_respiratorycharting_df["charttime"])
        
        data_respiratorycharting_df['value'] = data_respiratorycharting_df["value"].str.replace(r'\%', '')
        data_respiratorycharting_df['value'] = pd.to_numeric(data_respiratorycharting_df["value"])
        data_lab_df['unitid'] = pd.to_numeric(data_lab_df["unitid"])
        data_lab_df['charttime'] = pd.to_numeric(data_lab_df["charttime"])
        data_lab_df['value'] = pd.to_numeric(data_lab_df["value"])

        # Convert ICU centered time to time since hospital admission, so we have a unique time through all unitstays 
        data_respiratorycharting_df = self.preprocessor.create_time(data_respiratorycharting_df)
        data_lab_df = self.preprocessor.create_time(data_lab_df)
        
        # Create data frames for peep and FiO2
        peeplist = ['PEEP', 'PEEP/CPAP']
        fiO2list = ['FiO2', 'FiO2 (%)', 'O2 Percentage']
        data_peep_list = [data_respiratorycharting_df[data_respiratorycharting_df['name'] == name] for name in peeplist]
        data_peep_list.append(data_lab_df[data_lab_df['name']== 'PEEP'])
        data_peep_df = self._merge_df(data_peep_list)

        data_paO2_df = data_lab_df[data_lab_df['name']=='paO2']

        data_fiO2_list = [data_respiratorycharting_df[data_respiratorycharting_df['name'] == name] for name in fiO2list]
        data_fiO2_list.append(data_lab_df[data_lab_df['name']== 'FiO2'])        
        data_fiO2_df = self._merge_df(data_fiO2_list)
        
        
        
        
        
        
        # Calculate horowitz and timepoint of lowest horovitz-index
        horowitz_data, min_horowitz = self.preprocessor.calculate_Horowitz(data_fio2=data_fiO2_df, data_peep=data_peep_df, data_paO2= data_paO2_df, dict_adm_stays= dict_adm_stays, admissons=admissions)
        
        
        # Get relevant unitstays for each patient automaticly removes patients with no horowitz
        horowitz_hadm_ids_list = []
        start_time_list = []
        end_time_list = []

        # Get all unit stays of a patient in the relevant timeframe and ignore the rest
        for _, row in min_horowitz.iterrows():
            hadm_id = row['hadm_id']
            horowitz_hadm_ids_list.append(hadm_id)            
            unit_stays_list = dict_adm_stays[hadm_id]
            min_time = row['charttime']
            window_start_time = min_time - windowsize_data_before
            window_end_time = min_time + windowsize_data_after
            dict_adm_times[hadm_id] = (window_start_time, window_end_time)
            start_time_list.append(window_start_time)
            end_time_list.append(window_end_time)
            for stay in unit_stays_list:
                if dict_unit_admission[stay] <=  window_start_time and dict_unit_discharge[stay] > window_start_time:
                    relevant_stays_list.append(stay)
                    self.dict_stays_times[stay] = (window_start_time, window_end_time)
                elif dict_unit_admission[stay] > window_start_time and  dict_unit_admission[stay] <= window_end_time: 
                    relevant_stays_list.append(stay)
                    self.dict_stays_times[stay] = (window_start_time, window_end_time)
        
        # Update Dataframe with only patients with horovitz-index and the start and endtime of their respective window
        window_data['hadm_id'] = horowitz_hadm_ids_list
        window_data['starttime'] = start_time_list
        window_data['endtime'] = end_time_list

        # Delete stays without horowitz data from dictionaries
        for key in list(dict_adm_stays) :
            if key not in horowitz_hadm_ids_list:
                del dict_adm_stays[key]
        for key in list(dict_stay_index) :
            if key not in relevant_stays_list:
                del dict_stay_index[key]
                del dict_stays_adm[key]
                del dict_unit_admission[key]
                del dict_unit_discharge[key]
        list_stays = [str(stay) for stay in relevant_stays_list]
        list_adm_ids = horowitz_hadm_ids_list


        
        # Get relevant unit stays
        unit_stays_df = unit_stays_df.query('unitid in @relevant_stays_list')
        
        # Update unitstay id list of preprocessor
        self.preprocessor.update_unitstays(unit_stays_df)
        
        # Discard all admissions with no horovitz-index
        filter = eICUFilter(self.options)
        admissions = filter.filter_admissions(list_adm_ids, admissions)
        
        if len(list_stays) == 0:
            
            # Create empty pages if no valid patients are left
            page = self.preprocessor.generate_feature_page()
            freq = self.preprocessor.generate_variable_page()
            print(f"No valid patients {self.job_id}")

        else :

            # Discard data that is not in the relevant window for the respective patient
            data_fiO2_df = filter.filter_data_not_in_window(data_fiO2_df, self.dict_stays_times)
            data_paO2_df = filter.filter_data_not_in_window(data_paO2_df, self.dict_stays_times)
            data_peep_df = filter.filter_data_not_in_window(data_peep_df, self.dict_stays_times)
            horowitz_data = filter.filter_horowitz_not_in_window(horowitz_data, dict_adm_times)
            list_other_data = [data_fiO2_df, data_paO2_df, data_peep_df, horowitz_data]    
            
            # Extract aperiodic vital data
            query = f"""SELECT patientunitstayid as unitid, observationoffset as charttime,  paop, cardiacoutput, svri, pvri FROM vitalaperiodic WHERE patientunitstayid IN ({str(list_stays)[1:-1]}) AND NOT (paop = '' and cardiacoutput ='' and svri = '' and pvri = '');"""
            self.cursor.execute(query)
            data_vitalaperiodic = pd.DataFrame(self.cursor.fetchall(), columns=self.cursor.column_names)

            # Convert to correct data type
            data_vitalaperiodic = data_vitalaperiodic.replace('', np.nan)
            for column in data_vitalaperiodic:
                data_vitalaperiodic[column] = pd.to_numeric(data_vitalaperiodic[column])
            
            # Convert unit centered timestamps to admission centered timestamps
            data_vitalaperiodic = self.preprocessor.create_time(data_vitalaperiodic)

            # Discarded data not in the relevant window
            data_vitalaperiodic = filter.filter_data_not_in_window(data_vitalaperiodic ,self.dict_stays_times) 

            # Extract periodic vital data
            self.cursor.execute(f"""
                    SELECT patientunitstayid as unitid, observationoffset as charttime, temperature, saO2, heartrate, respiration, cvp, etco2, systemicsystolic, systemicdiastolic,  
                    systemicmean, pasystolic, padiastolic, pamean
                    FROM vitalperiodic WHERE patientunitstayid IN({str(list_stays)[1:-1]}) AND NOT (temperature = '' AND saO2 = '' AND heartrate = '' AND respiration = '' AND cvp = '' AND
                    etco2 = '' AND systemicsystolic = '' AND systemicdiastolic = '' AND systemicmean = '' AND pasystolic = '' AND padiastolic = '' AND pamean = '');
            """)
            data_vitalperiodic = pd.DataFrame(self.cursor.fetchall(), columns=self.cursor.column_names)
            data_vitalperiodic = data_vitalperiodic.replace('', np.nan)

            # Convert all data in dataframe to numeric values since there are stored as strings
            for column in data_vitalperiodic:
                data_vitalperiodic[column] = pd.to_numeric(data_vitalperiodic[column])
            data_vitalperiodic = filter.filter_data_not_in_window(data_vitalperiodic, self.dict_stays_times)

            list_other_data.append(data_vitalaperiodic)
            list_other_data.append(data_vitalperiodic) 

            # Variables needed for data extraction
            lab_names = list(mapping_extraction_lab_rev)
            list_labs = [[] for _ in range(len(lab_names))]
            repscharting_names = list(mapping_extraction_respcharting_rev)
            list_respcharting = [[] for _ in range(len(repscharting_names))]
            infusion_drug_lists = list(mapping_extraction_infusion_drug.values())
            list_infusion_drugs = [[] for _ in range(len(infusion_drug_lists))]

            # Extract relevant data from lab table and store resulting df in list (each relevant parameter has its on df)
            for item in lab_names :
                self.cursor.execute(f"""
                    SELECT patientunitstayid as unitid, labresultoffset as charttime, labresult value FROM lab WHERE labname = '{item}' and patientunitstayid IN ({str(list_stays)[1:-1]});
                """)
                data = pd.DataFrame(self.cursor.fetchall(), columns=self.cursor.column_names)
                
                # Fill empty cells with nan as they are the empty string by default and that leads to problems in processing
                data = data.replace('',np.nan)
                
                # Convert all necessary columns to numeric ones
                for column in data:
                    data[column] = pd.to_numeric(data[column])
                
                # Convert unitstay centered timestamp to admission centered timestamp and delete data out of relevant window
                data = self.preprocessor.clean_data(data, self.dict_stays_times)
                
                # Create empty dataframe if no data was recorded
                if len(data.index) == 0 :
                    data = pd.DataFrame(columns=['unitid','charttime', 'value'])
                list_labs[mapping_extraction_lab_rev.get(item)] = data

            # Extract relevant data from respiratorycharting table and store resulting df in list (each relevant parameter has its on df)
            for item in repscharting_names : 
                self.cursor.execute(f"""
                    SELECT patientunitstayid as unitid, respchartentryoffset as charttime, respchartvalue as value FROM respiratorycharting  WHERE  respchartvaluelabel = '{item}' and patientunitstayid IN ({str(list_stays)[1:-1]});
                """)
                data = pd.DataFrame(self.cursor.fetchall(), columns=self.cursor.column_names)
                
                # Fill empty cells with nan as they are the empty string by default and that leads to problems in processing
                data = data.replace('',np.nan)
                
                # Convert all necessary columns to numeric ones
                for column in data:
                    data[column] = pd.to_numeric(data[column])
                
                # Convert unitstay centered timestamp to admission centered timestamp and delete data out of relevant window
                data = self.preprocessor.clean_data(data, self.dict_stays_times)
                
                # Create empty df in case no data was recorded
                if len(data.index) == 0 :
                    data = pd.DataFrame(columns=['unitid','charttime', 'value'])
                list_respcharting[mapping_extraction_respcharting_rev.get(item)] = data

            # Extract relevant data from infusiondrugs table and store resulting df in list (each relevant drug has its on df)
            for item in infusion_drug_lists :
                self.cursor.execute(f"""
                    SELECT patientunitstayid as unitid, infusionoffset as charttime, drugname, drugrate, infusionrate, drugamount, volumeoffluid, patientweight FROM infusiondrug WHERE drugname IN
                    ({str(item)[1:-1]}) AND patientunitstayid in ({str(list_stays)[1:-1]});
                """)
                data = pd.DataFrame(self.cursor.fetchall(), columns=self.cursor.column_names)
                
                # Fill empty cells with nan as they are the empty string by default and that leads to problems in processing
                data = data.replace('',np.nan)

                # Convert all necessary columns to numeric ones
                for column in data:
                    if not column == "drugname" :
                        if column == 'drugrate':
                            cleaned = data[column].tolist()
                            for index in range(len(cleaned)):
                                cleaned[index] = re.sub('[^-0-9,.]+','', cleaned[index])
                                if cleaned[index].startswith("."):
                                    cleaned[index] = cleaned[index].replace('.', '0.')
                                    cleaned[index] = cleaned[index].replace(',', '')
                            
                            data[column] = cleaned
                            data[column] = pd.to_numeric(cleaned)
                            
                            
                        else : 
                            data[column] = pd.to_numeric(data[column])

                # Convert unitstay centered timestamp to admission centered timestamp and delete data out of relevant window
                data = self.preprocessor.clean_data(data, self.dict_stays_times)
                
                # Create empty df in case no data was recorded
                if len(data.index) == 0 :
                    data = pd.DataFrame(columns=['unitid','charttime', 'drugname', 'drugrate', 'infusionrate', 'drugamount', 'volumeoffluid', 'patientweight'])
                
                list_infusion_drugs[mapping_extraction_infusion_drug_rev.get(item[0])] = data
            
            # Convert data into feature vector and 
            page, freq = self.preprocessor.process_data(list_other_data, list_infusion_drugs, list_labs, list_respcharting,  window_data, dict_adm_stays,  admissions, min_horowitz)

            
        return page, freq

        

    def _merge_df(self, list_df: list) -> pd.DataFrame:
        """Function that takes a list of dfs merges them into one and calculates the mean of entries at the same time"""

        # Ensure unique name and value column names
        for index in range(len(list_df)) :
            name = 'name' + str(index)
            value = 'value' + str(index)

            list_df[index].rename(columns={'value': value, 'name' :name}, inplace=True)
        data = reduce(lambda  left,right: pd.merge(left,right,on=['charttime', 'unitid'], how='outer'), list_df)
        
        columns = []

        # Create dataframe only containing values
        for col in data.columns:
            if re.search('value', col):
                columns.append(col)
        column_data = data[columns]
        
        # Calculate the mean at each timestamp
        data['value'] = column_data.mean(axis=1)
        for col in columns:
            del data[col]
        return data


    
