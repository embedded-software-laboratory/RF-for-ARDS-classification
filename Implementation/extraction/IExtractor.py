
from typing import Any

import numpy as np
from numpy import  ndarray


from Lookup.diagnosis_indicies import diagnosis_dict
from Lookup.uka_extraction_indicies import *
from Lookup.all_features import *
from Lookup.features_new import *

class IExtractor:
    """This class is the base class of all extraction"""
    def __init__(self, options: Any, job_id: int):
        # Store options and job_id in class variable
        self.options = options
        self.job_id = job_id
        

    def extract_page(self, page_size: int, page_offset: int) -> ndarray:
        """Function used to extract Df that contains all relevant information. Each row is one patient."""
        pass

    
    @staticmethod
    def generate_empty_page(pagesize: int, rows: int) -> ndarray:
        """Function that provides a empty numpy array which is used for basic patient information"""
        array = np.empty((rows, pagesize), dtype=object)

        # Fill with empty arrays
        for i, row in enumerate(array):
            for j, _ in enumerate(row):
                array[i][j] = []

        return array
    
    

    