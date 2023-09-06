from typing import Any

import pandas as pd

from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, f_classif, SelectFromModel, SelectPercentile


from learning.ILearner import ILearner
from learning.sk_Learner import SKLearner


"""Class that is used to apply different feature Selection algorithms"""
class SKFeatureSelector(ILearner) :
    def __init__(self,  options: Any):
        super().__init__(options)
        self.options = options
        
        # Get necessary options for feature selection        
        self.feature_selection_options = options["feature_selection"]
        self.feature_selection_settings = self.feature_selection_options["parameters"]
        self.fs_alogrithm_parameters = self.feature_selection_options["algorithm_params"]
        self.learning_options = options["learning"]
        self.learning_locations = self.learning_options["locations"]
        self.sklearner = SKLearner(options)
        return

    def feature_selection(self):
        """Function that is used to apply feature selection"""

        # Determines if a new split of datasets should be used
        if self.feature_selection_settings["generate_new_sets"] == 1:
            self._generate_new_full_dataset()

        # Read Training data
        predictors, labels = self._read_data(location="../Data/Training_Data/dump_feature_selection.parquet")

        # Select feature selection algorithm WARNING non supported algorithms not implemented
        algorithm = None
        for item in  self.feature_selection_settings["algorithms"]:
            if item in ["chi2", "mutual_info_classif", "f_classif"] :
                match item:
                    case "chi2":
                        algorithm = chi2
                    case "mutual_info_classif" :
                        algorithm = mutual_info_classif
                    case "f_classif" :
                        algorithm = f_classif
                    case _:
                        print("Error unknow algorithm feature selcetion: " + item)
                if self.feature_selection_settings["type"] == "SelectKBest" :
                    # Train model and select best performing columns
                    model = SelectKBest(algorithm, k=self.fs_alogrithm_parameters["SelectKBest"]["k"])
                    model.fit(predictors, labels)
                    cols = model.get_support(indices=True)
                    predictors = predictors.iloc[:, cols]

                elif self.feature_selection_settings["type"] == "SelectPercentile":
                    # Train model and select best performing columns
                    model = SelectPercentile(algorithm, percentile = self.fs_alogrithm_parameters["SelectPercentile"]["percentile"])
                    model.fit(predictors, labels)
                    cols = model.get_support(indices=True)
                    predictors = predictors.iloc[:, cols]
                else :
                    print("Error unknow algorithm feature selcetion: " + self.feature_selection_settings["type"])
                
            elif item == "Tree" :
                tree_type = self.feature_selection_settings["type"]
                params = self.feature_selection_settings["algorithm_params"]["Tree"]
                clf = None
                match tree_type:
                    case "Random_Forest" :
                        clf = self.sklearner._init_forest(params["Random_Forest_location"])

                    case _:
                        print("Error unknow algorithm feature selcetion: " + tree_type)
                
                # Train model and select best performing columns
                clf = clf.fit(predictors, labels)
                model = SelectFromModel(clf, prefit=True)
                cols = model.get_support(indices=True)
                predictors = predictors.iloc[:, cols]
        
        # Write data with only best selected columns
        training_set_selected = predictors
        training_set_selected["ARDS"] = labels
        self._write_data(training_set_selected, self.learning_locations["training_set_location"])
        return

     

        


