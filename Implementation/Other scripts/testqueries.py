
from datetime import datetime
import datetime as dt
import json
import os

import re
from os.path import exists
import math

import numpy as np
import mysql.connector
import pandas as pd


from ssh import open_ssh
from extraction.MIMICIVExtractor import MIMICIVExtractor
from extraction.eICUExtractor import eICUExtractor
from extraction.ukaExtractor import ukaExtractor
from Lookup.uka_extraction_indicies import *
from Lookup.uka_data_indicies import uka_column_names



def read_options():
    with open("C:/Users/Haenf/OneDrive/Dokumente/Bachelorarbeit/ba-pieper/06_Implementierung/Implementation/options.json", "r") as file:
        return json.load(file)

def start_connection(_options, page_size: int, offset: int):

    connection = mysql.connector.connect(
            host = options["database_connection"]["host"],
            port = options["database_connection"]["port"],
            user = options["database_connection"]["user"],
            password = options["database_connection"]["password"],
            database = options["database_connection"]["database"],
            use_pure = True,
            ssl_disabled = True
        )
    match options["database_connection"]["database"]:
        case "SMITH_MIMICIV":
            print("MIMICIV")
            extractor = MIMICIVExtractor(_options, connection.cursor())
        case "SMITH_eICU":
            print("SMITH_eICU")
            extractor = eICUExtractor(_options, connection.cursor())
        case "SMITH_ASIC_SCHEME":
            print("uka")
            extractor = ukaExtractor(_options, connection.cursor())
        case _: 
            print("Something went wrong!")
            return
    
    extractor.extract_page(page_size, offset)
    
    
def test_data_extraction(_options, samples):
    connection = mysql.connector.connect(
            host = _options["host"],
            port = _options["port"],
            user = _options["user"],
            password = _options["password"],
            database = _options["database"],
            use_pure = True,
            ssl_disabled = True
        )
    windowsize_horowitz = 3
    cutoff_horowitz_time = 1440
    windowsize_data = 10080
    total_avg_list = []

    # Data extraction for every admission
    for admission in samples:
        admission_length = admission[ADMISSION_LENGTH]
        admission_length = admission_length[0]
        admission_id = admission[ADMISSION_ID]
        print(admission_id)
        cursor = connection.cursor()
        # Extract Horowitz values where PEEP is higher than 5cmH20
        cursor.execute("""
            SELECT `Horowitz-Quotient_(ohne_Temp-Korrektur)` as horowitz, Zeit_ab_Aufnahme as time  FROM uka_data 
                WHERE patientid = %s AND `Horowitz-Quotient_(ohne_Temp-Korrektur)` IS NOT NULL AND PEEP >= 5;
            """, admission_id
        )


        patient_horowitz = pd.DataFrame(cursor.fetchall())
        if patient_horowitz.shape[0]>0:
            patient_horowitz.columns = cursor.column_names
        else:
            continue

        #Calculates time differences between to different Horowitz entries
        patient_horowitz['time_diff'] = patient_horowitz['time'].rolling(2).apply(lambda x: x.iloc[1] - x.iloc[0])

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
        max_horowitz_entries = 0
        for df in split_patient_horowitz:
            length = len(df)
            if max_horowitz_entries< length:
                max_horowitz_entries = length
        if max_horowitz_entries<windowsize_horowitz:
            windowsize_horowitz = max_horowitz_entries
        for df in split_patient_horowitz:
            
            avg_horowitz.extend(df['horowitz'].rolling(windowsize_horowitz).mean())
        
        patient_horowitz['mean_horowitz'] = avg_horowitz
        # Get time of minimal horowitz
        
        idx_min_avg_horowitz = patient_horowitz['mean_horowitz'].idxmin()
        min_horowitz_time = patient_horowitz['time'][idx_min_avg_horowitz]
        
        # Determine Cutoff times for Data extraction based on the selected windowsize and lowest horowitz time
        max_data_time = (min_horowitz_time+windowsize_data) if (min_horowitz_time+windowsize_data) <= admission_length else admission_length
        min_data_time = (min_horowitz_time-windowsize_data) if (min_horowitz_time-windowsize_data) >= 0 else 0
        
        
        # Extract relevant data TODO rename columns
        cursor.execute(f"""
            SELECT * FROM uka_data 
            WHERE patientid = {str(admission_id[0])} AND Zeit_ab_Aufnahme BETWEEN {str(min_data_time)} AND {str(max_data_time)};
        """)
        
        patient_data = pd.DataFrame(cursor.fetchall())
        patient_data.columns = cursor.column_names
        days = (max_data_time-min_data_time)/1440
        patient_avg_list = []
        for name in uka_column_names:
            count = 0
            for i in range(patient_data.shape[0]):
                
                if patient_data.at[i,name] is not None and  not math.isnan(patient_data.at[i,name]):
                    count +=1
            
            temp = count/days
            patient_avg_list.append(temp)
            
        total_avg_list.append(patient_avg_list)
    
    df = pd.DataFrame(total_avg_list)
    avg = pd.DataFrame(df.sum(axis=0)/df.shape[0])
    np.savetxt('../Data/avg.txt', avg.values)
def extract_collumn_names(_options, table: any):
        connection = mysql.connector.connect(
            host = _options["host"],
            port = _options["port"],
            user = _options["user"],
            password = _options["password"],
            database = _options["database"],
            use_pure = True,
            ssl_disabled = True
        )

        cursor = connection.cursor()
        cursor.execute(f"""
            SELECT COLUMN_NAME
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_NAME = '{table}'
                ORDER BY ORDINAL_POSITION
            """
        )
        column_names = cursor.fetchall()
        if exists('../Data/uka_data_column_names.txt'):
            os.remove('../Data/uka_data_column_names.txt')
        with open(r'../Data/uka_data_column_names.txt', 'w') as fp:
            for name in column_names:
                fp.write("%s\n" % name)


def allicd10codes_querry(_options):
    connection1 = mysql.connector.connect(
            host = _options["host"],
            port = _options["port"],
            user = _options["user"],
            password = _options["password"],
            database = "SMITH_ASIC_SCHEME",
            use_pure = True,
            ssl_disabled = True
        )
    cursor = connection1.cursor()
    cursor.execute(""" 
        SELECT `ICD-10_Codes` FROM uka_patients;
    """)
    icd10codes = cursor.fetchall()
    print("Data received")
    list_codes = []
    
    for i in range(len(icd10codes)):
        split_string = icd10codes[i][0].split(",", 100)
        for j in range(len(split_string)):
            #here
            if split_string[j][0:3] not in list_codes:
                #here
                list_codes.append(split_string[j][0:3])
    
   

    print("After uka:" + str(len(list_codes)))
    
    connection2 = mysql.connector.connect(
            host = _options["host"],
            port = _options["port"],
            user = _options["user"],
            password = _options["password"],
            database = "SMITH_MIMICIV",
            use_pure = True,
            ssl_disabled = True
        )
    cursor = connection2.cursor()
    cursor.execute(
        """
        SELECT DISTINCT icd_code FROM diagnoses_icd WHERE icd_version = 10;
        """
    )
    icd10codes = cursor.fetchall()
    print("Data Received")
    

    for i in range(len(icd10codes)):
        code = icd10codes[i][0][0:3]
        if code not in list_codes:
            list_codes.append(code)
    
    print("After MIMIC:" + str(len(list_codes)))

    connection3 = mysql.connector.connect(
            host = _options["host"],
            port = _options["port"],
            user = _options["user"],
            password = _options["password"],
            database = "SMITH_eICU",
            use_pure = True,
            ssl_disabled = True
        )
    
    cursor = connection3.cursor()
    cursor.execute("""
        SELECT DISTINCT icd9code FROM diagnosis;
    """)
    icd10codes = cursor.fetchall()
    print("Data Reiceved")
    for i in range(len(icd10codes)):
        temp = icd10codes[i][0]
        temp = temp.replace(" ", "")
        if re.search('[a-zA-Z]', temp):
            codes = temp.split(",", 100)
            for j in range (len(codes)):
                if re.search('[a-zA-Z]', codes[j]):
                        if codes[j][0:3] not in list_codes:
                            list_codes.append(codes[j][0:3])
    print("After eICU:" + str(len(list_codes)))
    icd10 = True
    for j in range(len(list_codes)):
        #print(list_codes[j])
        if not list_codes[j][0].isalpha():
            icd10 = False
    if len(list_codes) == len(set(list_codes)):
        print("False")
    else:
        print("TRUE")
    print("ALL ICD10:" + str(icd10))
    
    list_codes.sort()
    if exists('../Data/allicd10codes.txt'):
        os.remove('../Data/allicd10codes.txt')
    with open(r'../Data/allicd10codes.txt', 'w') as fp:
        for code in list_codes:
           fp.write("%s\n" % code)
def filtericd10eIcu_querry(_options):
    connection3 = mysql.connector.connect(
            host = _options["host"],
            port = _options["port"],
            user = _options["user"],
            password = _options["password"],
            database = "uka",
            use_pure = True,
            ssl_disabled = True
        )
    
    cursor = connection3.cursor()
    cursor.execute("""
        SELECT * FROM (
               SELECT
                   p.uniquepid , p.gender , p.patienthealthsystemstayid , p.age as age,
                   unitdischargeoffset as admission_length FROM patient p
                   GROUP BY p.patienthealthsystemstayid
                   ORDER BY p.patienthealthsystemstayid
            ) AS d WHERE (d.age = '> 89' or d.age>=18) AND d.admission_length>= 120
               LIMIT 200 OFFSET 0; 
    """)
    
    admissions = cursor.fetchall()

    admission_ids = []
    for i in range(len(admissions)):
        admission_ids.append(admissions[i][2])
    
    cursor.execute(f"""
        SELECT c.patienthealthsystemstayid, d.icd9code  FROM diagnosis d 
            INNER JOIN (SELECT p.patientunitstayid, p.patienthealthsystemstayid FROM patient p 
            WHERE p.patienthealthsystemstayid in ({str(admission_ids)[1:-1]}) GROUP BY patientunitstayid) as c 
            ON c.patientunitstayid = d.patientunitstayid 
            ORDER BY c.patienthealthsystemstayid;"""
    )
    diagnosis = cursor.fetchall()
    
    for i in range(len(diagnosis)):
        if re.search('[a-zA-Z]', diagnosis[i][1]):
            try: 
                admission_ids.remove(diagnosis[i][0])
            except:
                pass
    admissions = [a for a in admissions if not a[2] in admission_ids ]
    print(len(admissions))
    print(type(admissions))
    print(admissions[0])
    
def TestI_E(_options):
    connection = mysql.connector.connect(
        host = _options["database_connection"]["host"],
        port = _options["database_connection"]["port"],
        user = _options["database_connection"]["user"],
        password = _options["database_connection"]["password"],
        database = "SMITH_MIMICIV",
        use_pure = True,
        ssl_disabled = True
    )
    cursor = connection.cursor()
    cursor.execute("""
        SELECT valuenum, itemid FROM SMITH_MIMICIV.chartevents cn WHERE itemid = 226871 or itemid = 226873 and  valuenum > 0;
    """)
    
    data = pd.DataFrame(cursor.fetchall())
    I_ERAtio = []
    min = float('inf')
    max = float('-inf')
    sum= 0
    print("data fetched")
    list_exp = []
    list_insp = []
    for _, row  in data.iterrows():
        if row[1] == 226871:
            list_exp.append(row)
        else:
            list_insp.append(row)
    insp = pd.concat(list_insp)
    exp = pd.concat(list_exp)
    #list_ratio = []
    for index1, item1 in insp.items():
        for index2, item2 in exp.items():
            if index1 == index2:
                print(item1/item2)
                break
    
def test_MIMIC_TIME_Extraction(_options):
    connection = mysql.connector.connect(
        host = _options["database_connection"]["host"],
        port = _options["database_connection"]["port"],
        user = _options["database_connection"]["user"],
        password = _options["database_connection"]["password"],
        database = "SMITH_MIMICIV",
        use_pure = True,
        ssl_disabled = True
    )
    cursor = connection.cursor()  
    cursor.execute("""
        SELECT charttime FROM SMITH_MIMICIV.chartevents cn LIMIT 120;
    """)  
    data = cursor.fetchall()
    minutes_diff = (data[119][0]-data[0][0]).total_seconds()/60.0
    print(minutes_diff)

if __name__ == "__main__":
    print("Reading options")
    options = read_options()

    # Open SSH connection for DB connection
    print("Opening SSH tunnel...")
    tunnel = open_ssh(options["database_connection"]["ssh"]["host"], options["database_connection"]["ssh"]["username"], options["database_connection"]["ssh"]["password"],
                             options["database_connection"]["ssh"]["forward_host"], options["database_connection"]["ssh"]["forward_port"])
    options["database_connection"]["host"] = "localhost"  # This makes extractors use the SSH tunnel
    options["database_connection"]["port"] = tunnel.local_bind_port
    options["database_connection"]["database"] = "SMITH_MIMICIV"
    
    

    JOB_ID = 1
    PAGE_SIZE = 400
    OFFSET = 9200
    #filtericd10eIcu_querry(options)
    
    #allicd10codes_querry(options)

    start_connection(options, PAGE_SIZE, OFFSET)

    #samples, times = extract_collumn_names(options, "uka_data")
    
    #TestI_E(options)
    #with open("D:\Test.txt","r") as file_avg_values:
    #    lines_avg_values = [float(line.rstrip().replace('\r','').replace('\n','')) for line in file_avg_values]
    #    min_entry = float('inf')
    #    max_entry = float('-inf')
    #    for entry in lines_avg_values:
    #        if entry < min_entry:
    #            min_entry = entry
    #        if entry > max_entry:
    #            max_entry = entry
    #    print(str(sum(lines_avg_values)/len(lines_avg_values)))
    #    print(max_entry)
    #    print(min_entry)
    #test_MIMIC_TIME_Extraction(options)

    tunnel.close()
    
