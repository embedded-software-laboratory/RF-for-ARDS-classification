from typing import Any

import pandas as pd

import numpy as np
from numpy import ndarray



from extraction.IExtractor import IExtractor
from preprocessing.ukaPreprocessor import ukaPreprocessor
from Lookup.uka_extraction_indicies import *

from Lookup.features_new import *
from Lookup.all_features import *

"""Class that is used to extract data from the database provided by the univerisity hospital Aachen"""
class ukaExtractor(IExtractor):
    
    def __init__(self, options: Any, cursor: Any, job_id:int):
        # Init class variables
        super().__init__(options, job_id)
        self.cursor = cursor
        self.preprocessor = ukaPreprocessor(self.options, self.job_id)
        
    
    def extract_page(self, pagesize: int, offset: int):
        """Function used to extract Df that contains all relevant information for the patient, the second return value contains 
        the total number of occurrences of the parameters. Each row in patient_data corresponds to one patient. Data is extracted for patients
        which are found in between rows offset and offset+pagesize of the patient table"""
        
        samples = IExtractor.generate_empty_page(pagesize, 9)
        
        
        # Get amount of admissions that are atleast 2h long with patients over 18 and an ICD-10 coded diagnosis
        admission_count = self._extract_admisions(samples,  pagesize, offset)
       

        # Return empty feature page and frequency page
        if admission_count == 0:
            print("\tDidn't find any admissions for ", (pagesize, offset))
            return self.preprocessor.generate_feature_page(), self.preprocessor.generate_variable_page()

        # Delete empty rows
        empty_row_count = pagesize - admission_count
        if empty_row_count > 0:
            samples = np.delete(samples, [t + admission_count for t in range(empty_row_count)], axis=1)
        # Get further data for patients
        patients_data, freq = self._extract_data(samples.transpose())    
        

       
        return patients_data, freq


    def _extract_admisions(self, samples: ndarray,  pagesize: int, offset: int):
        # Get Patients older than 18 and with a stay longer than 2h
        
        self.cursor.execute("""
            SELECT * FROM (
               SELECT
                   data.patientid,  up.clusterGeschlecht as gender, up.clusterAlter as admit_age, data.admission_length , up.`ICD-10_Codes`,
                   up.clusterKoerpergroesse as height, up.clusterKoerpergewicht as weight
               FROM uka_patients up 
                   INNER JOIN (
                   SELECT
                       MAX(ud.Zeit_ab_Aufnahme) as admission_length, ud.patientid
                   FROM  uka_data ud
                   GROUP BY ud.patientid
                   ) as data
                   ON data.patientid = up.patientid
            ) AS d WHERE d.admit_age >=18 and admission_length >=120
               LIMIT %s offset %s;""", (pagesize, offset)
            
        )
        admissions = self.cursor.fetchall()
        
        # Store basic information for each patient into one numpy array
        for i in range(len(admissions)):
            samples[SUBJECT_ID][i].append(admissions[i][0])
            samples[ADMISSION_ID][i].append(admissions[i][0])
            samples[GENDER][i].append(admissions[i][1])
            samples[AGE][i].append(admissions[i][2])
            samples[ADMISSION_LENGTH][i].append(admissions[i][3])
            samples[DIAGNOSIS][i] = admissions[i][4]
            samples[HEIGHT][i] = admissions[i][5]
            samples[WEIGHT][i] = admissions[i][6]
        
        
        return len(admissions)

    def _extract_data(self, samples: ndarray):
        """Function that extracts all relevant parameters for the patients that are contained in samples. It returns a dataframe with the features used 
        during the learning process as well as the total number of occurrences of the parameters"""
        
        
        #Dertermines timespan in minutes (before and after) that are taken into account from point of lowest horowitz
        #Note the -15 in windowsize_data after are caused by the fact that the timestamp of min horowitz already count as after occurence of ARDS
        days_before = self.options["extraction_parameter"]["windowsize_before"]
        days_after = self.options["extraction_parameter"]["windowsize_after"]
        windowsize_data_before = days_before * 24 * 60
        windowsize_data_after = days_after * 24 * 60 - 15
        patients_data = []
        patients_frequencies = []
        # Data extraction for every admission 
        
        to_delete = []
        
        # Iterate over all patients in samples to get their relevant data
        for admission in samples:
            
            admission_length = admission[ADMISSION_LENGTH]
            admission_length = admission_length[0]
            admission_id = admission[ADMISSION_ID]
            
            
            # Extract Horowitz values where PEEP is higher than 5cmH20
            self.cursor.execute("""
                SELECT `Horowitz-Quotient_(ohne_Temp-Korrektur)`, Zeit_ab_Aufnahme as time  FROM uka_data 
                    WHERE patientid = %s AND `Horowitz-Quotient_(ohne_Temp-Korrektur)` IS NOT NULL AND PEEP >= 5;
                """, admission_id
            )

            # Skip patient if no horovitz-index is found since it is needed further into the process
            patient_horowitz = pd.DataFrame(self.cursor.fetchall())
            if patient_horowitz.shape[0]>0:
                patient_horowitz.columns = self.cursor.column_names
            else:
                continue
                
            
            # Find the time where the horovitz-value is minimal
            min_horowitz_time, min_horowitz = self.preprocessor.calculateHorowitz(patient_horowitz)

            
            # Determine Cutoff times for Data extraction based on the selected windowsize and lowest horowitz time
            # Determine if time window is shorter than expected due to limited stay time
            max_data_time = min_horowitz_time+windowsize_data_after
            min_data_time = min_horowitz_time-windowsize_data_before
            if max_data_time > admission_length:
                max_data_extraction_time = admission_length
            else :
                max_data_extraction_time = max_data_time
                
                
                
            if min_data_time <  0:
                min_data_extraction_time = 0
            else :
                min_data_extraction_time = min_data_time    
            
            
            
            
            # Extract relevant data
            self.cursor.execute(f"""
                SELECT * FROM uka_data 
                WHERE patientid = {str(admission_id[0])} AND Zeit_ab_Aufnahme BETWEEN {str(min_data_extraction_time)} AND {str(max_data_extraction_time)};
            """)
            
            
            # Create DF containing all relevant data
            patient_db_data = pd.DataFrame(self.cursor.fetchall(), columns=self.cursor.column_names)
            
            # Process extracted data
            feature, freq = self.preprocessor.process_data(patient_db_data, admission, min_horowitz_time, max_data_time, min_data_time)
            patients_data.append(feature)
            patients_frequencies.append(freq)
        
        # Concat data of single patients into one dataframe where each row corresponds to one patient and return the result
        page_data = pd.concat(patients_data, ignore_index=True)
        page_freq = pd.concat(patients_frequencies, ignore_index=True)
        return page_data, page_freq


            



            

            
            

            
            

            

    
