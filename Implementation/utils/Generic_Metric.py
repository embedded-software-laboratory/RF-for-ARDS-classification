from typing import Any
import numpy as np

class GenericMetric:
    def __init__(self, threshold_calc: str, metric_name: str, split: Any, values: Any = None):
        self._threshold_calc = threshold_calc
        self._split = split
        self._name = metric_name

        if values is None:
            self._values = []
        else:
            if isinstance(values, np.ndarray):
                self._values = values.tolist()
            else:
                self._values = values

    @property
    def split(self):
        return self._split

    @split.setter
    def split(self, value):
        self._split = value

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, value):
        if isinstance(value, np.ndarray):
            self._values = value.tolist()
        else:
            self._values = value


    def to_dict(self) -> (str, dict):
        generic_metric_dict = {self.split: self._values}
        return self._name, self._threshold_calc, generic_metric_dict
