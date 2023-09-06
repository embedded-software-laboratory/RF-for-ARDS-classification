from typing import Any

from numpy import ndarray
import numpy as np
import pandas as pd

from Lookup.uka_extraction_indicies import *
# from extraction.diagnosis_indicies import diagnosis_dict
from filtering.IFilter import IFilter

"""Class that contains filters specific to the eICU database"""
class eICUFilter(IFilter):

    def __init__(self, options: Any) -> None:
        super().__init__(options)
        self.options = options

   
    def filter_admissions(self, list_ids: list, admissons: ndarray) -> ndarray:
        """Function that deletes all data from admissions whoose id is not in the provided list"""
        to_delete = []
        for index in range(len(admissons)):
            if not admissons[index][ADMISSION_ID] in list_ids:
                to_delete.append(index)
        cleaned = np.delete(admissons, to_delete, 0)
        return cleaned

    # 
    def filter_data_not_in_window(self, data: pd.DataFrame, dict_stay_times: dict) -> pd.DataFrame:
        """ Function that deletes all data which is not recorded in the window for the stay to which the data belongs"""
        
        to_delete = []
        data = data.reset_index()
        del data['index']
        for index, row in data.iterrows():
            # Check if unitid is relevant
            if row['unitid'] not in list(dict_stay_times):
                to_delete.append(index)
            
            # Check if charttime is in window of relevant data
            elif row['charttime'] < dict_stay_times.get(row['unitid'])[0] or row['charttime'] > \
                    dict_stay_times.get(row['unitid'])[1]:
                to_delete.append(index)
        data = data.drop(to_delete)
        data = data.reset_index()
        del data['index']
        return data

    def filter_horowitz_not_in_window(self, data: pd.DataFrame, dict_hadm_times: dict) -> pd.DataFrame:
        """ Function that deletes all horovitz indices which are not recorded in the window for the stay to which the data belongs"""
        
        data = data.reset_index()
        del data['index']
        

        to_delete = []
        for index, row in data.iterrows():
            # Check if admission is a relevant admission
            if row['hadm_id'] not in list(dict_hadm_times):
                to_delete.append(index)

            # Check if recorded charttime is within the relevant window
            elif row['charttime'] < dict_hadm_times.get(row['hadm_id'])[0] or row['charttime'] > \
                    dict_hadm_times.get(row['hadm_id'])[1]:
                to_delete.append(index)
        data = data.drop(to_delete)
        data = data.reset_index()#
        
        del data['index']
        return data

    def filter_data_admission(self, data: pd.DataFrame, unitstays: list) -> pd.DataFrame:
        """Function that discards all rows whoose unit id is not in unitstays"""
        return data.query("unitid in @unitstays")
