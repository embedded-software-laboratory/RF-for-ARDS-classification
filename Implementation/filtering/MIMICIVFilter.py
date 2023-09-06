from typing import Any
from datetime import datetime
import math

from Lookup.diagnosis_indicies import diagnosis_dict
from Lookup.uka_extraction_indicies import *
from filtering.IFilter import IFilter

from numpy import ndarray
import pandas as pd

"""Class that is used during processing to delete unnecessary data"""
class MIMICIVFilter(IFilter):

    def __init__(self, options: Any):
        super().__init__(options)
        


    
    
    
    def _delete_data_not_in_window(self, data: pd.DataFrame, start_data: datetime, end_data: datetime, admit_time: datetime):
        """Deletes all data that are not in the window and converts timestamps to minutes since admission"""    
        
        # Initilize variables
        index_start = math.nan
        index_end = math.nan
        rows_to_drop = []

        # Find first and last entry within 
        for i in range(len(data.index)):
            if data['charttime'].at[i]< start_data:
                index_start = i
                continue
            elif data['charttime'].at[i]> end_data:
                index_end = i
                break
            data['charttime'].at[i] = (data['charttime'].at[i] - admit_time).total_seconds() / 60.0
        
        # Drop unneeded rows
        if not math.isnan(index_start):
            rows_to_drop = rows_to_drop + [i for i in range(0, index_start+1, 1)]
        if not math.isnan(index_end):
            rows_to_drop = rows_to_drop + [i for i in range(index_end, data.last_valid_index()+1,1)]
        if not len(rows_to_drop) == 0:
            data = data.drop(rows_to_drop)
        
        # Ensure right datatypes in df
        if len(data.index) == 0:
            if 'value' in data.columns:
                data = pd.DataFrame(columns=['admission_id', 'itemid', 'charttime',  'value'])
                data = data.astype({'admission_id' : 'int64', 'itemid': 'int64', 'charttime': 'float64', 'value': 'float64'})
            elif 'mean_horowitz' in data.columns:
                data = pd.DataFrame(columns=['abstime', 'charttime',  'admission_id',  'Horowitz-Quotient_(ohne_Temp-Korrektur)',  'time_diff',  'mean_horowitz'])
                data = data.astype({'abstime' : 'float64', 'charttime': 'float64',  'admission_id':'int64',  'Horowitz-Quotient_(ohne_Temp-Korrektur)': 'float64',  'time_diff':'float64',  'mean_horowitz':'float64'})
        else :
            data['charttime'] = data['charttime'].astype('float64')
            data = data.reset_index()
            del data['index']
            data = data.infer_objects()
        
        
        return data

    def _delete_data_not_in_window_input(self, data: pd.DataFrame, start_data: datetime, end_data: datetime, admit_time: datetime):
        """Deletes all data that are not in the window and converts timestamps to minutes since admission"""    
        
        # Initilize variables
        index_start = math.nan
        index_end = math.nan
        # Find first index after window
        rows_to_drop = []
        for i in range(len(data.index)):
            if data['starttime'].at[i] > end_data:
                index_end = i
                break
            elif data['endtime'].at[i]> end_data and  (data['way'].at[i] == 'Bolus' or data['way'].at[i] == 'Drug Push'):
                data['endtime'].at[i] = data['starttime'].at[i]
            elif data['endtime'].at[i]> end_data and not (data['way'].at[i] == 'Bolus' or data['way'].at[i] == 'Drug Push'):
                time_running = data['endtime'].at[i] - data['starttime'].at[i]
                time_considered = end_data - data['starttime'].at[i]
                value_considered = data['value'].at[i]*(time_considered/time_running)
                data['value'].at[i] = value_considered
                data['endtime'].at[i] = end_data
                
        # Drop unwanted rows
        if not math.isnan(index_end):
            rows_to_drop = rows_to_drop + [i for i in range(index_end, data.last_valid_index()+1,1)]
        if not len(rows_to_drop)==0:
            data = data.drop(rows_to_drop)
        data = data.sort_values('endtime', ascending=True, kind='mergesort')
        data = data.reset_index()
        del data['index'] 

        # Find first index before window
        rows_to_drop = []
        
        for i in range(len(data.index)):
            if data['endtime'].at[i]< start_data:
                index_start = i
                continue
            elif data['starttime'].at[i]< start_data and  (data['way'].at[i] == 'Bolus' or data['way'].at[i] == 'Drug Push'):
                rows_to_drop.append(i)
            elif data['starttime'].at[i]< start_data and not (data['way'].at[i] == 'Bolus' or data['way'].at[i] == 'Drug Push'):
                time_running = data['endtime'].at[i] - data['starttime'].at[i]
                time_considered = data['endtime'].at[i] - start_data
                value_considered = data['value'].at[i]*(time_considered/time_running)
                data['value'].at[i] = value_considered
                data['starttime'].at[i] = start_data

        
            
        # Drop unwanted rows
        if not math.isnan(index_start):
            rows_to_drop = rows_to_drop + [i for i in range(0, index_start+1,1)]
            
        if not len(rows_to_drop)==0:
            data = data.drop(rows_to_drop)
            data = data.reset_index()
            del data['index']
         
        # Set time to minutes since admission
        for i in range(len(data.index)):
            data['starttime'].at[i] = (data['starttime'].at[i] - admit_time).total_seconds() / 60.0
            data['endtime'].at[i] = (data['endtime'].at[i] - admit_time).total_seconds() / 60.0
        
        # Ensure sorted df
        data = data.sort_values('starttime', ascending=True, kind='mergesort')
        data = data.reset_index()
        del data['index']

        # Ensure correct data types
        if len(data.index) == 0:
            data = pd.DataFrame(columns=['admission_id', 'itemid', 'starttime', 'endtime', 'value', 'rate', 'amountuom', 'rateuom', 'way', 'patientweight'])
            data = data.astype({'admission_id' : 'int64', 'itemid': 'int64', 'starttime': 'float64', 'endtime':'float64', 'value': 'float64',  'rate': 'float64',
                                 'amountuom': 'str', 'rateuom': 'str', 'way': 'str', 'patientweight': 'float64'})
        else :
            data['starttime'] = data['starttime'].astype('float64')
            data['endtime'] = data['endtime'].astype('float64')
            data = data.infer_objects()
        

        return data


    def _delete_data_not_in_window_procedure(self, data: pd.DataFrame, start_data: datetime, end_data: datetime, admit_time: datetime):
        """Deletes all data that are not in the window and converts timestamps to minutes since admission"""    
        
        # Initiliaze variables
        index_start = math.nan
        index_end = math.nan
        rows_to_drop = []

        # Find first index after window
        for i in range(len(data.index)):
            if data['starttime'].at[i] > end_data:
                index_end = i
                break
            
            elif data['endtime'].at[i]> end_data :
                time_running = data['endtime'].at[i] - data['starttime'].at[i]
                time_considered = end_data - data['starttime'].at[i]
                value_considered = data['value'].at[i]*(time_considered/time_running)
                data['value'].at[i] = value_considered
                data['endtime'].at[i] = end_data
                
        
        # Store unnecessary rows and drop them
        if not math.isnan(index_end):
            rows_to_drop = rows_to_drop + [i for i in range(index_end, data.last_valid_index()+1,1)]
            
        if not len(rows_to_drop)==0:
            data = data.drop(rows_to_drop)
        
        # Ensure correct order
        data = data.sort_values('endtime', ascending=True, kind='mergesort')
        data = data.reset_index()
        del data['index'] 
        
        # Find first index before window
        rows_to_drop = []
        for i in range(len(data.index)):
            if data['endtime'].at[i]< start_data:
                index_start = i
                continue
            
            elif data['starttime'].at[i]< start_data :
                time_running = data['endtime'].at[i] - data['starttime'].at[i]
                time_considered = data['endtime'].at[i] - start_data
                value_considered = data['value'].at[i]*(time_considered/time_running)
                data['value'].at[i] = value_considered
                data['starttime'].at[i] = start_data

        # Store unnecessary rows and drop them   
        if not math.isnan(index_start):
            rows_to_drop = rows_to_drop + [i for i in range(0, index_start+1,1)]
            
        if not len(rows_to_drop)==0:
            data = data.drop(rows_to_drop)
        
        # Ensure correct index
        data = data.reset_index()
        del data['index'] 

        # Set time to minutes since admission
        for i in range(len(data.index)):
            data['starttime'].at[i] = (data['starttime'].at[i] - admit_time).total_seconds() / 60.0
            data['endtime'].at[i] = (data['endtime'].at[i] - admit_time).total_seconds() / 60.0
        data = data.sort_values('starttime', ascending=True, kind='mergesort')
        data = data.reset_index()
        del data['index']
        
        # Make df right datatype
        data = data.infer_objects()
        data['starttime'] = data['starttime'].astype('float64')
        data['endtime'] = data['endtime'].astype('float64')
        return data


    def _delete_data_not_in_window_ventilation(self, data: pd.DataFrame, start_data: datetime, end_data: datetime, admit_time: datetime) :
        """Deletes all data that are not in the window and converts timestamps to minutes since admission"""

        # Init variables 
        data_filtered = pd.DataFrame(columns=['starttime', 'endtime'])
        start_list = []
        end_list = []

        # Find all relevant rows
        for index, row in data.iterrows():
            if row['end'] < start_data:
                data.drop([index])
                continue
            elif row['start'] > end_data : 
                data.drop([index])
                continue
            if row['start'] < start_data:
                start_list.append((start_data - admit_time).total_seconds() / 60.0)
            if row['start'] >= start_data :
                start_list.append((row['start'] - admit_time).total_seconds() / 60.0)
            if row['end'] > end_data:
                end_list.append((end_data-admit_time).total_seconds()/60.0)
            if row['end'] <=  end_data:
                end_list.append((row['end']-admit_time).total_seconds()/60.0)
    
        # Create Dataframe
        data_filtered['start'] = start_list
        data_filtered['end'] = end_list
        
        
        return data_filtered.sort_values(by='start', ascending=True)