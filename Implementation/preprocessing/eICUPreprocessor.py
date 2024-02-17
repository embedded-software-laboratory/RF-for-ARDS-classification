import math
from typing import Any
from functools import reduce

from numpy import ndarray
import pandas as pd
import numpy as np

from Lookup.MIMICIV_item_dict import *
from Lookup.Feature_mapping_eICU import *

from Lookup.uka_extraction_indicies import *
from preprocessing.IPreprocessor import IPreprocessor
from filtering.eICUFilter import eICUFilter
from Lookup.features_new import *
from Lookup.static_information import *


"""Class that is used to preprpcess Data extracted from the Extractor. The class provides various functions already used during extraction.
The most important function is process_data which takes the extracted data and returns the feature vector"""
class eICUPreprocessor(IPreprocessor):
    def __init__(self, options: Any, unitstays: pd.DataFrame, job_id: int):
        super().__init__(options, job_id)
        self.options = options
        self.update_unitstays(unitstays)
        
    
    def update_unitstays(self, unitstays: pd.DataFrame) :
        """Function that stores relevant unit stay ids in a class variable"""
        self.unitstays = unitstays
        self.dict_stay_time = {}
        
        for _, row in self.unitstays.iterrows():
            self.dict_stay_time[row['unitid']] = row['unitadmittime24']
        

    def process_data(self, list_other_data: list, list_drugs: list, list_lab: list, list_respcharting: list, window_data: pd.DataFrame, dict_adm_stays: dict,  
                        admissions: ndarray, min_horowitz_timestamps: pd.DataFrame) -> tuple:
        """Function that processes all parameters and creates a dataframe which has values on all parameters at all times. If no information is known for a timestamp its value is NaN.
         Information on admission, as well times important for the windows is needed"""
        
        list_data = [[] for _ in range(len(mapping_preprocessing_variabel_index))]
        
        # Transform data that is present as multiple variables in the DB into one variable
        data_fio2 = list_other_data[0]
        data_paO2 = list_other_data[1]
        data_peep = list_other_data[2]
        data_horowitz = list_other_data[3]
        vitalaperiodic = list_other_data[4]
        vitalperiodic = list_other_data[5]
        
        # Get Respiratory Rate
        vital_respiration = vitalperiodic[['unitid', 'charttime', 'respiration']].dropna(subset=['respiration'])
        list_RR = [vital_respiration, list_respcharting[mapping_extraction_respcharting_rev.get('TOTAL RR')],
                    list_lab[mapping_extraction_lab_rev.get('Respiratory Rate')]]
        data_RR = self.process_RR(list_RR) 
        
        # Get spontanous Resp. Rate
        list_RR_spont = [list_respcharting[mapping_extraction_respcharting_rev.get('RR Spont')], list_respcharting[mapping_extraction_respcharting_rev.get('Spontaneous Respiratory Rate')], 
                        list_respcharting[mapping_extraction_respcharting_rev.get('RR (patient)')], list_lab[mapping_extraction_lab_rev.get('Spontaneous Rate')]]
        data_RR_spont = self.process_spont_RR(list_RR_spont)
        
        # Get Tidalvolume
        list_TV = [list_respcharting[mapping_extraction_respcharting_rev.get('Tidal Volume Observed (VT)')], list_respcharting[mapping_extraction_respcharting_rev.get('Exhaled Vt')],
                    list_respcharting[mapping_extraction_respcharting_rev.get('Exhaled TV (machine)')], list_respcharting[mapping_extraction_respcharting_rev.get('Exhaled TV (patient)')]]
        data_TV = self.process_TV(list_TV)

        # Get spontanous Tidalvolume
        list_TV_spont = [list_respcharting[mapping_extraction_respcharting_rev.get('Spont TV')], list_respcharting[mapping_extraction_respcharting_rev.get('Vt Spontaneous (mL)')], 
                        list_respcharting[mapping_extraction_respcharting_rev.get('Exhaled TV (patient)')]]
        data_TV_spont = self.process_TV_spont(list_TV_spont)

        # Get compliance
        list_compliance = [list_respcharting[mapping_extraction_respcharting_rev.get('Static Compliance')],list_respcharting[mapping_extraction_respcharting_rev.get('Compliance')]]
        data_compliance = self.process_Compliance(list_compliance)
        
        # GET endinspiratory pressure
        list_P_EI = [list_respcharting[mapping_extraction_respcharting_rev.get('Plateau Pressure')],list_respcharting[mapping_extraction_respcharting_rev.get('Peak Insp. Pressure')],
                    list_lab[mapping_extraction_lab_rev.get('Peak Airway/Pressure')]]
        data_P_EI = self.process_P_EI(list_P_EI)

        # Get endtidal CO2 
        vital_etCO2 = vitalperiodic[['unitid', 'charttime', 'etco2']].dropna(subset=['etco2'])
        list_ETCO2 = [vital_etCO2, list_respcharting[mapping_extraction_respcharting_rev.get('EtCO2')]]
        data_ETCO2 = self.process_EtCO2(list_ETCO2)
        data_ETCO2 = self._etCO2(data_ETCO2)
        
        # Convert values to correct units
        list_data[mapping_extraction_preprocessing_lab.get('albumin')] = self._albumin(list_lab[mapping_extraction_lab_rev.get('albumin')])
        list_data[mapping_extraction_preprocessing_lab.get('total bilirubin')] = self._Bilirubin_total(list_lab[mapping_extraction_lab_rev.get('total bilirubin')])
        list_data[mapping_extraction_preprocessing_lab.get('Hgb')] = self._Hemoglobin(list_lab[mapping_extraction_lab_rev.get('Hgb')])
        list_data[mapping_extraction_preprocessing_lab.get('BUN')] = self._urea_nitrogen(list_lab[mapping_extraction_lab_rev.get('BUN')])
        list_data[mapping_extraction_preprocessing_lab.get('creatinine')] = self._creatinine(list_lab[mapping_extraction_lab_rev.get('creatinine')])   
        list_data[mapping_extraction_preprocessing_lab.get('CRP')] = self._CRP(list_lab[mapping_extraction_lab_rev.get('CRP')], 95.238)
        
        # Store data (without drugs) in one list for easier acess
        list_data[mapping_preprocessing_variabel_index.get('AF')] = data_RR
        list_data[mapping_preprocessing_variabel_index.get('AF_spontan')] = data_RR_spont
        list_data[mapping_preprocessing_variabel_index.get('Compliance')] = data_compliance
        list_data[mapping_preprocessing_variabel_index.get('DAP')] = vitalperiodic[['charttime', 'unitid', 'systemicdiastolic']].dropna(subset=['systemicdiastolic']).rename(columns={'systemicdiastolic' : 'value'})
        list_data[mapping_preprocessing_variabel_index.get('HF')] = vitalperiodic[['charttime', 'unitid', 'heartrate']].dropna(subset=['heartrate']).rename(columns={'heartrate' : 'value'})
        list_data[mapping_preprocessing_variabel_index.get('Horowitz-Quotient_(ohne_Temp-Korrektur)')] = data_horowitz.rename(columns={'Horowitz-Quotient_(ohne_Temp-Korrektur)': 'value'})
        list_data[mapping_preprocessing_variabel_index.get('Koerperkerntemperatur')] = vitalperiodic[['charttime', 'unitid', 'temperature']].dropna(subset=['temperature']).rename(columns={'temperature' : 'value'})
        list_data[mapping_preprocessing_variabel_index.get('MAP')] = vitalperiodic[['charttime', 'unitid', 'systemicmean']].dropna(subset=['systemicmean']).rename(columns={'systemicmean' : 'value'})
        list_data[mapping_preprocessing_variabel_index.get('PEEP')] = data_peep
        list_data[mapping_preprocessing_variabel_index.get('Tidalvolumen')] = data_TV
        list_data[mapping_preprocessing_variabel_index.get('P_EI')] = data_P_EI
        list_data[mapping_preprocessing_variabel_index.get('SAP')] = vitalperiodic[['charttime', 'unitid', 'systemicsystolic']].dropna(subset=['systemicsystolic']).rename(columns={'systemicsystolic' : 'value'})
        # According to eICU documentation data on SaO2 in the vitalperiodic table is collected by peripheral monitoring thus we store it as SpO2
        list_data[mapping_preprocessing_variabel_index.get('SpO2')] = vitalperiodic[['charttime', 'unitid', 'saO2']].dropna(subset=['saO2']).rename(columns={'saO2' : 'value'})
        list_data[mapping_preprocessing_variabel_index.get('ZVD')] = vitalperiodic[['charttime', 'unitid', 'cvp']].dropna(subset=['cvp']).rename(columns={'cvp' : 'value'})
        list_data[mapping_preprocessing_variabel_index.get('paO2_(ohne_Temp-Korrektur)')] = data_paO2
        list_data[mapping_preprocessing_variabel_index.get('SVRI')] = vitalaperiodic[['charttime', 'unitid', 'svri']].dropna(subset=['svri']).rename(columns={'svri' : 'value'})
        list_data[mapping_preprocessing_variabel_index.get('HZV_(kontinuierlich)')] = vitalaperiodic[['charttime', 'unitid', 'cardiacoutput']].dropna(subset=['cardiacoutput']).rename(columns={'cardiacoutput' : 'value'})
        list_data[mapping_preprocessing_variabel_index.get('MPAP')] = vitalperiodic[['charttime', 'unitid', 'pamean']].dropna(subset=['pamean']).rename(columns={'pamean' : 'value'})
        list_data[mapping_preprocessing_variabel_index.get('DPAP')] = vitalperiodic[['charttime', 'unitid', 'padiastolic']].dropna(subset=['padiastolic']).rename(columns={'padiastolic' : 'value'})
        list_data[mapping_preprocessing_variabel_index.get('SPAP')] = vitalperiodic[['charttime', 'unitid', 'pasystolic']].dropna(subset=['pasystolic']).rename(columns={'pasystolic' : 'value'})
        list_data[mapping_preprocessing_variabel_index.get('PVRI')] = vitalaperiodic[['charttime', 'unitid', 'pvri']].dropna(subset=['pvri']).rename(columns={'pvri' : 'value'})
        list_data[mapping_preprocessing_variabel_index.get('FiO2')] = data_fio2
        list_data[mapping_preprocessing_variabel_index.get('Tidalvolumen_spont')] = data_TV_spont
        list_data[mapping_preprocessing_variabel_index.get('CK')] = list_lab[mapping_extraction_lab_rev.get('CPK')]
        list_data[mapping_preprocessing_variabel_index.get('GOT')] = list_lab[mapping_extraction_lab_rev.get('AST (SGOT)')]
        list_data[mapping_preprocessing_variabel_index.get('GPT')] = list_lab[mapping_extraction_lab_rev.get('ALT (SGPT)')]
        list_data[mapping_preprocessing_variabel_index.get('Haematokrit')] = list_lab[mapping_extraction_lab_rev.get('Hct')]
        list_data[mapping_preprocessing_variabel_index.get('INR')] = list_lab[mapping_extraction_lab_rev.get('PT - INR')]
        list_data[mapping_preprocessing_variabel_index.get('Kreatinin')] = list_lab[mapping_extraction_lab_rev.get('creatinine')]
        list_data[mapping_preprocessing_variabel_index.get('Laktat_arteriell')] = list_lab[mapping_extraction_lab_rev.get('lactate')]
        list_data[mapping_preprocessing_variabel_index.get('Leukozyten')] = list_lab[mapping_extraction_lab_rev.get('WBC x 1000')]
        list_data[mapping_preprocessing_variabel_index.get('Lipase_MIMIC')] = list_lab[mapping_extraction_lab_rev.get('lipase')]
        list_data[mapping_preprocessing_variabel_index.get('Thrombozyten')] = list_lab[mapping_extraction_lab_rev.get('platelets x 1000')]
        list_data[mapping_preprocessing_variabel_index.get('pTT')] = list_lab[mapping_extraction_lab_rev.get('pTT')]
        list_data[mapping_preprocessing_variabel_index.get('paCO2_(ohne_Temp-Korrektur)')] = list_lab[mapping_extraction_lab_rev.get('paCO2_(ohne_Temp-Korrektur)')]
        list_data[mapping_preprocessing_variabel_index.get('etCO2')] = data_ETCO2
        list_data[mapping_preprocessing_variabel_index.get('BNP_MIMIC')] = list_lab[mapping_extraction_lab_rev.get('BNP')]
        list_data[mapping_preprocessing_variabel_index.get('Lymphozyten_prozentual')] = list_lab[mapping_extraction_lab_rev.get('-lymphs')]
        list_data[mapping_preprocessing_variabel_index.get('Troponin')] = list_lab[mapping_extraction_lab_rev.get('troponin - T')]
        list_data[mapping_preprocessing_variabel_index.get('Amylase')] = list_lab[mapping_extraction_lab_rev.get('amylase')]
        list_data[mapping_preprocessing_variabel_index.get('CK-MB_MIMIC')] = list_lab[mapping_extraction_lab_rev.get('CPK-MB')]
        list_data[mapping_preprocessing_variabel_index.get('LDH_MIMIC')] = list_lab[mapping_extraction_lab_rev.get('LDH')]
        list_data[mapping_preprocessing_variabel_index.get('I:E')] = list_lab[mapping_extraction_respcharting_rev.get('PD I:E RATIO')]
        list_data[mapping_preprocessing_variabel_index.get('PCWP')] = vitalaperiodic[['charttime', 'unitid', 'paop']].dropna(subset=['paop']).rename(columns={'paop' : 'value'})
        list_data[mapping_preprocessing_variabel_index.get('individuelles_Tidalvolumen_pro_kg_idealem_Koerpergewicht')] = list_lab[mapping_extraction_respcharting_rev.get('TV/kg IBW')]
        list_data[mapping_preprocessing_variabel_index.get('SaO2')] = list_lab[mapping_extraction_respcharting_rev.get('SaO2')]
        
        # Add drug data to list for easier access
        for index in range(len(list_drugs)):
            drugname = mapping_extraction_infusion_drug.get(index)[0]
            list_data[mapping_extraction_preprocessing_drugs.get(drugname)] = list_drugs[index]
        
        # Lists that contains feature data and frequency of features
        page = []
        freq = []

        # Process feature data for each admission
        for index in range(len(admissions)):
            admission = admissions[index]
            admission_id = int(str(admission[ADMISSION_ID])[1:-1])

            # Get timestamp of min horovitz index
            min_horowitz_timestamp = min_horowitz_timestamps.query('hadm_id == @admission_id')['charttime'].values[0]
            static_info = self._assign_static_information(admission, min_horowitz_timestamp, other_code_reason)
            
            list_data_admission= [[] for _ in range(len(mapping_preprocessing_variabel_index))]
            
            
            unitstays = dict_adm_stays.get(admission_id)
            
            # Only use information with the correct unitstayid
            for i in range(len(list_data)):
                df = list_data[i]
                if len(df)>=1 :
                    
                    
                    if 'hadm_id' in df.columns :
                        list_data_admission[i] = df[df['hadm_id'] == admission_id ]
                    else :
                        list_data_admission[i] = list_data[i].query('unitid in @unitstays')
                else :
                    list_data_admission[i] = pd.DataFrame(columns=['charttime', 'unitid', 'value'])
                
           
            times = window_data[window_data['hadm_id']==admission_id]
            times.reset_index(inplace=True)
            del times['index']
            # Process information from database into different features
            
            page_admission, freq_admission = self.process_data_admission(list_data_admission, times, static_info)
            
           
            page.append(page_admission)
            freq.append(freq_admission)
            
        return pd.concat(page), pd.concat(freq)
    
    def process_data_admission(self, list_data: list, times: pd.DataFrame, static_info: list) -> pd.DataFrame:
        """Function that process data for a single admission and prepares the creation of a full feature vector"""
        
        # Build timetable for whole window
        fulltimes_list = [x for x in np.arange(times['starttime'].at[0], times['endtime'].at[0], 1.0)]
        fulltimes_df = pd.DataFrame(columns=['charttime'])
        variable = self.generate_variable_page()
        fulltimes_df['charttime'] = fulltimes_list
        variable['Zeitpunkt'] = fulltimes_list

        # Calculate missing data and process infused drugs
        
        for index in range(len(list_data)):
            if index == mapping_preprocessing_variabel_index.get('deltaP'):
                list_data[index] = self._calculate_delta_P(list_data[mapping_preprocessing_variabel_index.get('PEEP')], list_data[mapping_preprocessing_variabel_index.get('P_EI')])

            elif index == mapping_preprocessing_variabel_index.get('Lymphozyten_absolut') :
                list_data[index] = self._calculate_Lymphs_abs(list_data[mapping_preprocessing_variabel_index.get('Lymphozyten_prozentual')], list_data[mapping_preprocessing_variabel_index.get('Leukozyten')])
            elif index in mapping_drug_index_used_unit:
                
                list_data[index] = self._process_drug_data(list_data[index], mapping_drug_index_used_unit.get(index), mapping_drug_index_needed_unit.get(index), static_info[static_dict.get('Gewicht')])

            
            # Create variable df with entry at every timestamp in the relevant window and assign paramters to their respective column
            data = fulltimes_df.merge(list_data[index], how='left', on='charttime')
            variable[mapping_preprocessing_variabel_index_rev.get(index)] = data['value']   

        variable.sort_values(by='Zeitpunkt', inplace=True)
        variable.reset_index(inplace=True)
        del variable['index']
        feature, freq = self._assign_features(variable, static_info, feature_position_dict_eICU_MIMIC)

        return feature, freq
    
    def _calculate_delta_P(self, data_PEEP: pd.DataFrame, data_PEI: pd.DataFrame) -> pd.DataFrame:
        """Function that calculates the pressuere difference between PEEP and PEI"""
        
        # Ensure unique value column names
        data_PEEP =data_PEEP.rename(columns={'value': 'PEEP'})
        data_PEI = data_PEI.rename(columns={'value' : 'PEI'})

        # Merge dData
        merged = data_PEEP.merge(data_PEI, how='outer', on='charttime')
        merged = merged.sort_values(by='charttime')
        merged = merged.reset_index()
        del merged['index']

        # Calculate difference
        deltap_times, deltap_values = super()._calculate_deltaP(merged, 'PEEP', 'PEI', 'charttime')
        
        # Create DF
        deltap = pd.DataFrame(columns=['charttime', 'value'])
        deltap['charttime'] = deltap_times
        deltap['value'] = deltap_values
        return deltap
        
    
    def _calculate_Lymphs_abs(self, data_Lymphs_percent: pd.DataFrame, data_leukos: pd.DataFrame) -> pd.DataFrame:
        """Function that calculates the absolute number of Lymphocytes"""
        
        # Ensure unique value column names
        data_Lymphs_percent = data_Lymphs_percent.rename(columns={'value' : 'lymphs'})
        data_leukos = data_leukos.rename(columns={'value' : 'leukos'})

        # Data merge
        merged = data_leukos.merge(data_Lymphs_percent, how='outer', on='charttime')
        lymphs_values = []
        lymphs_times = []
        merged = merged.sort_values(by='charttime')
        merged = merged.reset_index()
        del merged['index']

        # Calculate absolute count of Lymphocytes
        for index, row in merged.iterrows():
            # Find non nan values for percentage of lymphocytes
            if not math.isnan(row['lymphs']):
                time = row['charttime']
                for i in range(index, 0, -1):
                    # Find first non nan number of leukocytes recoreded before the percentage of lymphocytes
                    if not math.isnan(merged['leukos'].at[i]) and (time-merged['charttime'].at[i])<self.options["extraction_parameter"]["cutoff_time_lymphs_leukos"]:
                        # Calculate abs. number of lymphocytes
                        lymphs_times.append(row['charttime'])
                        lymphs_values.append(row['lymphs']*merged['leukos'].at[i])
                    break
        
        # Create Dataframe
        lymphs = pd.DataFrame(columns=['charttime', 'value'])
        lymphs['charttime'] = lymphs_times
        lymphs['value'] = lymphs_values

        return lymphs

    def calculate_Horowitz(self, data_fio2: pd.DataFrame, data_peep: pd.DataFrame, data_paO2: pd.DataFrame, dict_adm_stays: dict, admissons: ndarray) :
        
        horowitz_data_list = []
        min_horowitz_data_list = []
        horowitz_data = pd.DataFrame(columns=['hadm_id','charttime', 'Horowitz-Quotient_(ohne_Temp-Korrektur)'])
        min_horowitz_data = pd.DataFrame(columns=['hadm_id', 'charttime'])
        for admission in admissons:
            admission_id = admission[ADMISSION_ID][0]
            horowitz_unitstays = dict_adm_stays.get(admission_id)
            
            data_fio2_patient = data_fio2.query('unitid in @horowitz_unitstays')
            data_paO2_patient = data_paO2.query('unitid in @horowitz_unitstays')
            data_peep_patient = data_peep.query('unitid in @horowitz_unitstays')

            data_horowitz_patient = pd.DataFrame(columns=['hadm_id','charttime', 'Horowitz-Quotient_(ohne_Temp-Korrektur)'])
            
            if len(data_peep_patient.index) == 0 or len(data_fio2_patient.index) == 0 or len(data_fio2_patient.index) == 0:
                continue
            else :
                data_peep_patient = data_peep_patient.rename(columns={'value': 'PEEP'})
                data_fio2_patient = data_fio2_patient.rename(columns={'value': 'FiO2'})
                data_paO2_patient = data_paO2_patient.rename(columns={'value': 'paO2_(ohne_Temp-Korrektur)'})
            
                data_frames = [ data_fio2_patient[['charttime', 'FiO2']], data_paO2_patient[['charttime', 'paO2_(ohne_Temp-Korrektur)']], data_peep_patient[['charttime', 'PEEP']]]
                horowitz_params_full  = reduce(lambda  left,right: pd.merge(left,right,on=['charttime'], how='outer'), data_frames).sort_values(by='charttime')
                patient_horowitz_value_list = []
                patient_horowitz_time_list = []
                
                for index, row in horowitz_params_full.iterrows():
                    if not math.isnan(row['paO2_(ohne_Temp-Korrektur)']):
                        for i in range(index, 0, -1):
                            
                            if not self._check_ventilated(admission, row['charttime'], horowitz_params_full['charttime'].at[i]):
                                break

                            if not math.isnan(horowitz_params_full['FiO2'].at[i]) and not horowitz_params_full['FiO2'].at[i] == 0:
                                if not self._check_PEEP(row, horowitz_params_full, patient_horowitz_value_list, patient_horowitz_time_list, i) :
                                    break
                data_horowitz_patient['charttime'] = patient_horowitz_time_list
                data_horowitz_patient['Horowitz-Quotient_(ohne_Temp-Korrektur)'] = patient_horowitz_value_list
                data_horowitz_patient['hadm_id'] = [admission_id for _ in range(len(patient_horowitz_time_list))]
                if len(data_horowitz_patient.index) > 0:
                    horowitz_data_list.append(data_horowitz_patient)
                    min_horowitz_data_list.append(self._find_min_Horowitz(data_horowitz_patient, admission_id))
 
        if not len(horowitz_data_list) == 0:
        
            horowitz_data = pd.concat(horowitz_data_list)
            min_horowitz_data = pd.concat(min_horowitz_data_list)
        else :
            horowitz_data = pd.DataFrame(columns=['charttime', 'Horowitz-Quotient_(ohne_Temp-Korrektur)', 'hadm_id'])
            min_horowitz_data = pd.DataFrame(columns=['charttime', 'hadm_id'])
        return horowitz_data, min_horowitz_data

    def predict_ventilation(self, data: pd.DataFrame, admission_ids: list) -> dict:
        """Function that predict possible ventilation timeframes by taking into account the presence of parameters correlated with mechanical ventilation"""
        
        ventinfo_dict = dict()
        
        # Iterate over all admissions
        for adm_id in admission_ids:

            # Get all unitstay ids for the current admission
            relevant_stays = self.unitstays.query('patienthealthsystemstayid == @adm_id')
            
            # Iterate over each unit stay
            for index, stay in relevant_stays.iterrows():
                stay_id = stay['unitid']

                # Extract data for the current unit stay
                relevant_data = data.query('unitid == @stay_id')
                relevant_data.reset_index(inplace=True)
                del relevant_data['index']

                # Calculate ventilation times
                ventinfo = self._calculate_ventilation_times(relevant_data)
                ventinfo_dict[adm_id] = ventinfo
        return ventinfo_dict
    
    def _calculate_ventilation_times(self, data_ventilation: pd.DataFrame) -> list:
        """Function that calculates start and endtime as well as duration of possible timeframes where mechanical ventliation might be present"""

        times = []

        # Get all relevant times and add them to a list and sort them to be ascending
        for _, row in data_ventilation.iterrows():
            charttime = row['charttime'] + self.dict_stay_time.get(row['unitid'])
            times.append(charttime)
        
        times = list(dict.fromkeys(times))
        times.sort()
        last_time = np.nan
        start_time = np.nan

        ventilation_start = []
        ventilation_end = []
        ventilation_duration = []        
        cutoff = 240

        # Check time difference between recorded parameters, if the difference is lower than 240 minutes it is likely that it is the same ventilation timeframe.
        for time in times :
            if math.isnan(start_time):
                start_time = time

            # Store start, endtime and duration of a ventilation window
            if not math.isnan(last_time):
                if (time - last_time) > cutoff:
                    ventilation_start.append(start_time)
                    ventilation_end.append(last_time)
                    ventilation_duration.append(last_time-start_time)
                    start_time = time
            last_time = time
        return [ventilation_start, ventilation_end, ventilation_duration]
    
  
    def create_time(self,  data: pd.DataFrame) -> pd.DataFrame:
        """Function that transforms the unitstay centered timestamps into admission centered timestamps"""

        for index, row in data.iterrows():
            data['charttime'].at[index] = self.dict_stay_time.get(row['unitid']) + row['charttime']
        data = data.sort_values('charttime')
        data = data.reset_index()
        del data['index'] 
        return data

    # Converts unit centered timestamps into admission centered timestamps and removes data that is out of the patients window
    def clean_data(self,  data: pd.DataFrame, dict_stays_times: dict) -> pd.DataFrame :
        filter = eICUFilter(self.options)
        data = self.create_time(data)
        data  = filter.filter_data_not_in_window(data, dict_stays_times)
        return data

    def _find_min_Horowitz(self, data_horowitz: pd.DataFrame, hadm_id: int) -> pd.DataFrame:
        data_horowitz.rename(columns={'charttime' : 'abstime', 'horowitz' : 'Horowitz-Quotient_(ohne_Temp-Korrektur)'}, inplace=True)
        
        idx_min_avg_horowitz, min_horowitz = super().find_lowest_Horowitz(data_horowitz)
        data_horowitz.rename(columns={'abstime' : 'charttime'}, inplace=True)
        min_horowitz_charttime = data_horowitz['charttime'][idx_min_avg_horowitz]

        min_horowitz = pd.DataFrame(columns=['hadm_id',  'charttime'])
        min_horowitz['hadm_id'] = [hadm_id]
        min_horowitz['charttime'] = [min_horowitz_charttime]
        return min_horowitz
        

    
    # Check if next FiO2 is needed
    def _check_PEEP(self, row: Any, data: pd.DataFrame, patient_horowitz_value_list: list, patient_horowitz_charttime_list: list, index: int):
        
        
        # Check if PEEP nearest to fiO2 is > 5
        for j in range(index, 0, -1):
            if not math.isnan(data['PEEP'].at[j]) and data['PEEP'].at[j]< 5:
                return False
            elif not math.isnan(data['PEEP'].at[j]):
                
                # Calculate and store horowitz
                patient_horowitz_value_list.append(row['paO2_(ohne_Temp-Korrektur)']/(data['FiO2'].at[index]/100))
                patient_horowitz_charttime_list.append(row['charttime'])
                return False
        return True

    
            
    
    def process_spont_RR(self, list_RR_spont: list) -> pd.DataFrame:
        # Split df into one df for each measurement and ensure unqiue column names
        list_RR_spont[0] = list_RR_spont[0].rename(columns={'value' : 'RRspont'})
        list_RR_spont[1] = list_RR_spont[1].rename(columns={'value' : 'spontRR'})
        list_RR_spont[2] = list_RR_spont[2].rename(columns={'value' : 'patient'})
        list_RR_spont[3] = list_RR_spont[3].rename(columns={'value' : 'labSpontRR'})

        # Merge Data
        data = reduce(lambda  left,right: pd.merge(left,right,on=['charttime', 'unitid'], how='outer'), list_RR_spont)
        values = []
        ids = []
        times = []
        at_least_one = False


        # Since the origin of the lab spont RR is not known we only use it if no other spont RR is present
        for _, row in data.iterrows():
            
            if not (math.isnan(row['patient']) and math.isnan(row['RRspont']) and math.isnan(row['spontRR'])):
                values.append(row[['patient', 'RRspont', 'spontRR']].mean())
                times.append(row['charttime'])
                ids.append(row['unitid'])
                at_least_one = True
            if not math.isnan(row['labSpontRR']) :
                values.append(row['labSpontRR'])
                times.append(row['charttime'])
                ids.append(row['unitid'])
                at_least_one = True
            if not at_least_one:
                times.append(row['charttime'])
                ids.append(row['unitid'])
                values.append(np.nan)
        return self._make_df(times, ids, values)


    def process_RR(self, list_RR: list) -> pd.DataFrame:
        """Function that is used to merge the different values for respiratory rate into one parameter"""
        
        # Ensure unqiue column names
        list_RR[1] = list_RR[1].rename(columns={'value' : 'respRRvalue'})
        list_RR[2] = list_RR[2].rename(columns={'value' : 'labRRvalue'})

        # Merge both dataframes
        data = reduce(lambda  left,right: pd.merge(left,right,on=['charttime', 'unitid'], how='outer'), list_RR)
        data.sort_values(by=['charttime'], inplace=True)
        times = []
        values = []
        ids = []
        at_least_one = False

        # Prepare creation of df
        for _, row in data.iterrows():
            if not math.isnan(row['respRRvalue']):
                values.append(row['respRRvalue'])
                times.append(row['charttime'])
                ids.append(row['unitid'])
                at_least_one = True
            if not math.isnan(row['respiration']):
                values.append(row['respiration'])
                times.append(row['charttime'])
                ids.append(row['unitid'])
                at_least_one = True
            if not math.isnan(row['labRRvalue']) :
                times.append(row['charttime'])
                ids.append(row['unitid'])
                values.append(row['labRRvalue'])
                at_least_one = True
            if not at_least_one :
                values.append(np.nan)
                times.append(row['charttime'])
                ids.append(row['unitid'])
        return self._make_df(times, ids, values)
    
    
    def process_TV(self, list_Tv: list) -> pd.DataFrame:
        """Function that is used to convert the different values for Tidalvolume into one column"""

        # Ensure unique column names
        list_Tv[0] = list_Tv[0].rename(columns={'value': 'observed'})
        list_Tv[1] = list_Tv[1].rename(columns={'value': 'exhaled'})
        list_Tv[2] = list_Tv[2].rename(columns={'value': 'machine'})
        list_Tv[3] = list_Tv[3].rename(columns={'value': 'patient'})

        # Merge data
        data = reduce(lambda  left,right: pd.merge(left,right,on=['charttime', 'unitid'], how='outer'), list_Tv)
        data.sort_values(by=['charttime'], inplace=True)
        
        # prepare creation of DF
        times = []
        values = []
        ids = []
        at_least_one = False
        for _, row in data.iterrows():
            if not math.isnan(row['observed']) :
                values.append(row['observed'])
                times.append(row['charttime'])
                at_least_one = True
                ids.append(row['unitid'])
            if not math.isnan(row['exhaled']) :
                values.append(row['exhaled'])
                times.append(row['charttime'])
                ids.append(row['unitid'])
                at_least_one = True
            if not math.isnan(row['machine']) and not math.isnan(row['patient']) :
                values.append(row['machine'] + row['patient'])
                times.append(row['charttime'])
                ids.append(row['unitid'])
                at_least_one = True
            if not math.isnan(row['machine']):
                values.append(row['machine'])
                times.append(row['charttime'])
                ids.append(row['unitid'])
                at_least_one = True
            if not math.isnan(row['patient']) :
                values.append(row['patient'])
                times.append(row['charttime'])
                ids.append(row['unitid'])
                at_least_one = True
            if not at_least_one :
                values.append(np.nan)
                times.append(row['charttime'])
                ids.append(row['unitid'])
        return self._make_df(times, ids, values)
    
    def process_TV_spont(self, list_TV_spont: list) -> pd.DataFrame:
        """Function that is used to convert the different values for spontanous Tidalvolume into one column"""

        # Ensure unique value column names
        list_TV_spont[0] = list_TV_spont[0].rename(columns={'value': 'spont'})
        list_TV_spont[1] = list_TV_spont[1].rename(columns={'value': 'ml'})
        list_TV_spont[2] = list_TV_spont[2].rename(columns={'value': 'patient'})

        # Merge data
        data = reduce(lambda  left,right: pd.merge(left,right,on=['charttime', 'unitid'], how='outer'), list_TV_spont)
        data.sort_values(by=['charttime'], inplace=True)

        # Prepare Creation of DF
        times = []
        values = []
        ids = []
        at_least_one = False
        for _, row in data.iterrows():
            if not math.isnan(row['spont']) :
                values.append(row['spont'])
                times.append(row['charttime'])
                ids.append(row['unitid'])
                at_least_one = True
            if not math.isnan(row['ml']) :
                values.append(row['ml'])
                times.append(row['charttime'])
                ids.append(row['unitid'])
                at_least_one = True
            if not math.isnan(row['patient']) :
                values.append(row['patient'])
                times.append(row['charttime'])
                ids.append(row['unitid'])
                at_least_one = True
            if not at_least_one :
                values.append(np.nan)
                times.append(row['charttime'])
                ids.append(row['unitid'])
        finished = self._make_df(times, ids, values)
        return finished

    def process_P_EI(self, list_P_EI: list) -> pd.DataFrame:
        """Function that is used to convert the different values for endinspiratory pressure into one column"""
        
        # Ensure unique value column names
        list_P_EI[0] = list_P_EI[0].rename(columns={'value': 'pPlateau'})
        list_P_EI[1] = list_P_EI[1].rename(columns={'value': 'pInsp'})
        list_P_EI[2] = list_P_EI[2].rename(columns={'value': 'peakPressure'})

        # Merge Data
        data = reduce(lambda  left,right: pd.merge(left,right,on=['charttime', 'unitid'], how='outer'), list_P_EI)
        data.sort_values(by=['charttime'], inplace=True)
        
        # Prepare Creation of DF
        times = []
        values = []
        ids = []
        for _, row in data.iterrows():
            if not math.isnan(row['pPlateau']):
                values.append(row['pPlateau'])
                times.append(row['charttime'])
                ids.append(row['unitid'])
            elif not math.isnan(row['pInsp']):
                values.append(row['pInsp'])
                times.append(row['charttime'])
                ids.append(row['unitid'])
            elif not math.isnan(row['peakPressure']):
                values.append(row['peakPressure'])
                times.append(row['charttime'])
                ids.append(row['unitid'])
            else :
                values.append(np.nan)
                times.append(row['charttime'])
                ids.append(row['unitid'])
        return self._make_df(times, ids, values)

    def process_Compliance(self, list_compliance: list) -> pd.DataFrame:
        """Function that is used to convert the different values for lung compliance into one column"""

        # Ensure unique value column names
        list_compliance[0] = list_compliance[0].rename(columns={'value': 'static'})
        list_compliance[1] = list_compliance[1].rename(columns={'value': 'dynamic'})
        data = reduce(lambda  left,right: pd.merge(left,right,on=['charttime', 'unitid'], how='outer'), list_compliance)
        data.sort_values(by=['charttime'], inplace=True)
        
        # Prepare creation of DF
        times = []
        values = []
        ids = []

        # After consulation with medical expert it was decided to use static compliance before dynamic compliance
        for _, row in data.iterrows():
            if not math.isnan(row['static']):
                values.append(row['static'])
                times.append(row['charttime'])
                ids.append(row['unitid'])
            elif not math.isnan(row['dynamic']):
                values.append(row['dynamic'])
                times.append(row['charttime'])
                ids.append(row['unitid'])
            else :
                values.append(np.nan)
                times.append(row['charttime'])
                ids.append(row['unitid'])
        return self._make_df(times, ids, values)
    
    def process_EtCO2(self, list_EtCO2: list) -> pd.DataFrame:
        """Function that is used to convert the different values for endtidal CO2 into one column"""
        
        # Ensure proper column names
        list_EtCO2[1]=list_EtCO2[1].rename(columns={'value' : 'ETCO2'})

        # Merge data
        data = reduce(lambda  left,right: pd.merge(left,right,on=['charttime', 'unitid'], how='outer'), list_EtCO2)
        data.sort_values(by=['charttime'], inplace=True)
        times = []
        values = []
        ids = []

        # Prepare creation of df
        for _, row in data.iterrows():
            times.append(row['charttime'])
            ids.append(row['unitid'])
            values.append(row[['ETCO2', 'etco2']].mean())
            
        return self._make_df(times, ids, values)
    
    
    def _process_drug_data(self, data: pd.DataFrame, list_units: str, to_unit: str, weight: float) -> pd.DataFrame:
        """Function processes drug data for storing in the feature vector"""
        
        # Preprocessing of drug data
        data = self._prepare_drug_data(data, list_units, to_unit, weight)
        data = data.rename(columns={'drugrate' : 'value'})
        
        if not len(data.index) == 0:
            data_full = self._impute_find_indexes(data=data, value_column='value', time_column='charttime', cutoff = 240, intervall = 1, filler=np.nan)
            
        else :
            data_full = data
        return data_full
        



  
    def _prepare_drug_data(self, data: pd.DataFrame, list_units: str, to_unit: str, weight: float) -> pd.DataFrame:
        """  Function that prepares drug data for next step by converting the drugs rates to the needed unit and omiting all unneccessary info"""
        
        
        data_list = []
        # Add empty df in case of no info
        if(len(data.index)) == 0:
            data_list.append(pd.DataFrame(columns=['charttime',  'drugrate']))
        else :
            for _, row in data.iterrows():
                
                # Get unit of measurement for stored drug application
                for i in range(len(list_units)):
                    unit = list_units[i]

                    # Check if unit is in the list of units that can be converted
                    if unit in row['drugname'] :
                        
                        # Determine right conversion
                        row_processed = self._determine_conversion(row, unit, to_unit, weight)

                        # Omit unnecessary information
                        if not row_processed.empty:
                            row_finished = row_processed.drop(['unitid','drugname', 'infusionrate', 'drugamount', 'volumeoffluid', 'patientweight'])
                            data_list.append(pd.DataFrame(row_finished).transpose())
        
        
        # Create new df out of processed drug data
        data_processed = pd.concat(data_list).reset_index()
        del data_processed['index']
        return data_processed

    
    def _determine_conversion(self, data: pd.DataFrame, from_unit: str, to_unit: str, weight: float) -> pd.DataFrame:
        """Function that determines the right conversion for drugrates, start and end unit must be given"""

        # Check if drugrate is recorded
        if math.isnan(data['drugrate']) :
            row_processed = pd.Series(dtype=object)
        else :
            if not math.isnan(data['patientweight']) :
                weight = data['patientweight']
            
            # Choose right dictionary of endunits to select the conversion from and convert drug rate
            match(to_unit[0]):
                case '(mcg/kg/min)':
                    row_processed = self.drug_conversion_dict_mcg_kg_min[from_unit](self, data, weight)
                case '(units/min)' :
                    row_processed = self.drug_conversion_dict_units_min[from_unit](self, data, weight)
                case '(mg/hr)' :
                    row_processed = self.drug_conversion_dict_mg_h[from_unit](self, data, weight)
                case _ :
                    print(to_unit[0])
        return row_processed
    
    
    # Function used for drugrate conversion
    def _convert_from_mcg_min_to_mcg_kg_min(self, data: pd.DataFrame, weight: float) -> pd.DataFrame:
        if math.isnan(weight) or weight == float(0) or math.isnan(data['drugrate']):
            data['drugrate'] = np.nan
        else :
            data['drugrate'] = data['drugrate']/weight
        return data

    def _convert_from_mg_kg_min_to_mcg_kg_min(self, data: pd.DataFrame, weight) -> pd.DataFrame:
        data['drugrate'] = data['drugrate'] * 1000
        return data

    def _convert_from_mcg_hr_to_mcg_kg_min(self, data: pd.DataFrame, weight: float) -> pd.DataFrame:
        if math.isnan(weight) or weight == float(0) or math.isnan(data['drugrate']):
            data['drugrate'] = np.nan
        else :
            data['drugrate'] = (data['drugrate']/60)/weight
        return data

    def _convert_from_mg_hr_to_mcg_kg_min(self, data: pd.DataFrame, weight: float) -> pd.DataFrame:
        if math.isnan(weight) or weight == float(0) or math.isnan(data['drugrate']):
            data['drugrate'] = np.nan
        else :
            data['drugrate'] = ((data['drugrate']*1000)/60)/weight
        return data

    def _convert_from_mg_min_to_mcg_kg_min(self, data: pd.DataFrame, weight: float) -> pd.DataFrame:
        if math.isnan(weight) or weight == float(0) or math.isnan(data['drugrate']):
            data['drugrate'] = np.nan
            pd.Series(index=data.index)
        else :
            data['drugrate'] = (data['drugrate']*1000)/weight
        return data

    def _convert_from_mcg_kg_hr_to_mcg_kg_min(self, data: pd.DataFrame, weight: float) -> pd.DataFrame:
        data['drugrate'] = data['drugrate']/60
        return data

    def _convert_from_mg_kg_hr_to_mcg_kg_min(self, data: pd.DataFrame, weight: float) -> pd.DataFrame:
        data['drugrate'] = data['drugrate']*1000/60
        return data

    def _convert_from_mcg_kg_min_to_mcg_kg_min(self, data: pd.DataFrame, weight: float) -> pd.DataFrame:
        return data

    def _convert_from_units_hr_to_units_min(self, data: pd.DataFrame, weight: float) -> pd.DataFrame:
        
        
        data['drugrate'] = data['drugrate']/60
        return data

    def _convert_from_units_kg_hr_to_units_min(self, data: pd.DataFrame, weight: float) -> pd.DataFrame:
        
        
        if math.isnan(weight) or weight == float(0) or math.isnan(data['drugrate']):
            data['drugrate'] = np.nan
        else :
            data['drugrate'] = data['drugrate']*weight/60
        return data

    def _convert_from_units_kg_min_to_units_min(self, data: pd.DataFrame, weight: float) -> pd.DataFrame:
        
        
        if math.isnan(weight) or weight == float(0) or math.isnan(data['drugrate']):
            data['drugrate'] = np.nan
        else :
            data['drugrate'] = data['drugrate']*weight
        return data

    def _convert_from_units_min_to_units_min(self, data: pd.DataFrame, weight: float) -> pd.DataFrame:
        return data

    def _convert_from_mcg_kg_min_to_mg_h(self, data: pd.DataFrame, weight: float) -> pd.DataFrame:
        data['drugrate'] = data['drugrate']*60/1000
        return data

    def _convert_from_mg_kg_hr_to_mg_h(self, data: pd.DataFrame, weight: float) -> pd.DataFrame:
        if math.isnan(weight) or weight == float(0) or math.isnan(data['drugrate']):
            data['drugrate'] = np.nan
        else :
            data['drugrate'] = data['drugrate']*weight
        return data

    def _convert_from_mcg_hr_to_mg_h(self, data: pd.DataFrame, weight: float) -> pd.DataFrame:
        data['drugrate'] = data['drugrate']/1000
        return data

    def _convert_from_mcg_kg_hr_to_mg_h(self, data: pd.DataFrame, weight: float) -> pd.DataFrame:
        if math.isnan(weight) or weight == float(0) or math.isnan(data['drugrate']):
            data['drugrate'] = np.nan
        else :
            data['drugrate'] = data['drugrate']*weight/1000
        return data

    def _convert_from_mcg_min_to_mg_h(self, data: pd.DataFrame, weight: float) -> pd.DataFrame:
        data['drugrate'] = data['drugrate']*60
        return data

    def _convert_from_mg_hr_to_mg_h(self, data: pd.DataFrame, weight: float) -> pd.DataFrame:
        return data

    def _convert_from_mg_kg_min_to_mg_h(self, data: pd.DataFrame, weight: float) -> pd.DataFrame:
        if math.isnan(weight) or weight == float(0) or math.isnan(data['drugrate']):
            data['drugrate'] = np.nan
        else :
            data['drugrate'] = data['drugrate']*60*weight
        return data

    def _convert_from_mg_min_to_mg_h(self, data: pd.DataFrame, weight: float) -> pd.DataFrame:
        data['drugrate'] = data['drugrate']*60
        return data

    
    
    def _make_df(self, times:list, ids: list, values: list) -> pd.DataFrame:
        """Function that creates a dataframe out of timestamp, value and unitstay id information"""

        processed_data = pd.DataFrame(columns=['charttime', 'unitid', 'value'])
        if not len(ids) == 0:
            processed_data['unitid'] = ids
        if not len(times) == 0:
            processed_data['charttime'] = times
        if not len(values) == 0:
            processed_data['value'] = values
        return processed_data

    
    # Dictoniaries for easier conversion
    drug_conversion_dict_mcg_kg_min={
        '(mcg/min)' : _convert_from_mcg_min_to_mcg_kg_min,
        '(mg/kg/min)' : _convert_from_mg_kg_min_to_mcg_kg_min,
        '(mcg/hr)' : _convert_from_mcg_hr_to_mcg_kg_min,
        '(mg/hr)' : _convert_from_mg_hr_to_mcg_kg_min,
        '(mg/min)' : _convert_from_mg_min_to_mcg_kg_min,
        '(mcg/kg/hr)' : _convert_from_mcg_kg_hr_to_mcg_kg_min,
        '(mg/kg/hr)' : _convert_from_mg_kg_hr_to_mcg_kg_min,
        '(mcg/kg/min)' : _convert_from_mcg_kg_min_to_mcg_kg_min,
    }
    drug_conversion_dict_units_min={
        '(units/hr)' : _convert_from_units_hr_to_units_min,
        '(units/kg/hr)' : _convert_from_units_kg_hr_to_units_min,
        '(units/kg/min)' : _convert_from_units_kg_min_to_units_min,
        '(units/min)' : _convert_from_units_min_to_units_min,
    }
    drug_conversion_dict_mg_h={
        '(mcg/kg/min)' : _convert_from_mcg_kg_min_to_mg_h,
        '(mg/kg/hr)' : _convert_from_mg_kg_hr_to_mg_h,
        '(mcg/hr)' : _convert_from_mcg_hr_to_mg_h,
        '(mcg/kg/hr)' : _convert_from_mcg_kg_hr_to_mg_h,
        '(mcg/min)' : _convert_from_mcg_min_to_mg_h,
        '(mg/hr)' : _convert_from_mg_hr_to_mg_h,
        '(mg/kg/min)' : _convert_from_mg_kg_min_to_mg_h,
        '(mg/min)' : _convert_from_mg_min_to_mg_h,
        '(mg/kg/min)' : _convert_from_mg_kg_min_to_mg_h,
    }
