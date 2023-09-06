


from typing import Any
import numpy as np
import pandas as pd

import pickle
import json

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV



from learning.ILearner import ILearner

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, matthews_corrcoef, roc_curve, make_scorer





class SKLearner(ILearner) :
    def __init__(self, options: Any):
        super().__init__(options)
        self.options = options
        self.learning_options = options["learning"]
        self.learning_params_general = self.learning_options["parameters"]["general"]
        self.learning_params_sk = self.learning_options["parameters"]["sk-learn"]
        self.random_forest_params = self.learning_params_sk["random_forest"]
        self.cross_validation_params = self.learning_params_sk["cross_validation"]
        self.learning_locations = self.learning_options["locations"]
        return
        

    def learn(self):
        """Function that starts the learning process of the RF and stores the resulting model after completion"""
        
        # Init forest and read training data
        random_forest = self._init_forest(self.random_forest_params)
        predictors, label = self._read_training_data()

        # Learn and store resulting model
        print("In learn")
        random_forest = random_forest.fit(predictors, label)
        self._store_model(random_forest)
        
        
        return

    @staticmethod
    def custom_scorer(labels, predictions) -> float:
            return matthews_corrcoef(labels, predictions)
    
    def cross_validate_forest(self) :
        """Function that perfroms crossvalidation"""
        
        
        
        settings_name = self.learning_params_general["model_name"]
        settings_path = self.learning_locations["validation_options_path"] + settings_name + ".json"
        metrics_path = self.learning_locations["validation_metrics_path"] + settings_name + ".json"

        # Load settings for RF
        file =  open(settings_path, "r")
        rf_settings = json.load(file)
        print(rf_settings)
        print(metrics_path)
        file.close()

        # Initialize forrest and read data
        random_forest = self._init_forest(rf_settings)
        predictors, labels = self._read_training_data()

        # Create Splits for Crossvalidation
        cross_validation = StratifiedKFold(n_splits=self.cross_validation_params["n_splits"], shuffle=self.cross_validation_params["shuffle"], random_state=None)
    
        # Variables for perfromance metrics
        fprs, tprs, scores, sensitivities, specificities, accuracies, mccs, f1s = [], [], [], [],[], [], [], []
        
        # Perform crossvalidation
        for (train_set, test_set), i in zip(cross_validation.split(predictors, labels), range(self.cross_validation_params["n_splits"])):
            predictors_train = predictors.iloc[train_set]
            labels_train = labels.iloc[train_set]
            
            # Learn model for the splut
            fitted = random_forest.fit(predictors_train, labels_train)

            # Evaluation of model
            _, _, auc_score_training = self._compute_roc_auc_cv(train_set, fitted, predictors, labels) 
            fpr, tpr, auc_score_test = self._compute_roc_auc_cv(test_set, fitted, predictors, labels) 
            f1, acc, mcc, sensitivity, specificity = self._compute_metrics_cv(test_set, fitted, predictors, labels)
            scores.append((auc_score_training, auc_score_test))
            fprs.append(fpr.tolist())
            tprs.append(tpr.tolist())
            sensitivities.append(sensitivity)
            specificities.append(specificity)
            accuracies.append(acc)
            
            mccs.append(mcc)
            f1s.append(f1)
        mean_acc = sum(accuracies)/len(accuracies)
        mean_sens = sum(sensitivities)/len(sensitivities)
        mean_spec = sum(specificities) / len(specificities)
        mean_f1 = sum(f1s)/ len(f1s)
        mean_mcc = sum(mccs)/len(mccs)
        metric_dict = {
            'tprs' : tprs,
            'fprs' : fprs,
            'auc_scores' : scores,
            'acc' : mean_acc, 
            'sens' : mean_sens,
            'spec' : mean_spec,
            'f1' : mean_f1,
            'mcc' : mean_mcc
        }
        file = open(metrics_path, 'w')
        json.dump(metric_dict, file)
        file.close()




    def evaluate(self) :
        """Function used to evalute a RF model"""

        # Load model and test data
        random_forest = self._load_model()
        predictors, labels = self._read_test_data()
        metrics_path = "../Data/Results/" + self.learning_params_general["model_name"] + ".json"
        
        # Compute evaluation results
        fpr, tpr, auc_score = self._compute_roc_auc(random_forest, predictors, labels) 
        f1, acc, mcc, sensitivity, specificity = self._compute_metrics(random_forest, predictors, labels)

        # Store results in file
        metric_dict = {
            'tprs' : tpr.tolist(),
            'fprs' : fpr.tolist(),
            'auc_scores' : auc_score,
            'acc' : acc, 
            'sens' : sensitivity,
            'spec' : specificity,
            'f1' : f1,
            'mcc' : mcc
        }
        file = open(metrics_path, 'w')
        json.dump(metric_dict, file)
        file.close()

        return 

    def predict_proba(self) :
        random_forest = self._load_model()
        predictors, label = self._read_training_data()
        result = random_forest.predict_proba(predictors, label)
        return result

    
    def _init_forest(self, location_rf_options) -> RandomForestClassifier:
        """Function that intializes the Random Forest"""
        
        # Check if new split into Training and Validation set should be performed
        if self.learning_params_general["generate_new_sets"] == 1:
            self._generate_new_full_dataset()

        # Init RF
        random_forest = RandomForestClassifier(
                            n_estimators = location_rf_options["n_estimators"],
                            criterion = location_rf_options["criterion"],
                            max_depth = location_rf_options["max_depth"],
                            min_samples_split = location_rf_options["min_samples_split"],
                            min_samples_leaf = location_rf_options["min_samples_leaf"],
                            min_weight_fraction_leaf = location_rf_options["min_weight_fraction_leaf"],
                            max_features = location_rf_options["max_features"],
                            max_leaf_nodes = location_rf_options["max_leaf_nodes"],
                            min_impurity_decrease =  location_rf_options["min_impurity_decrease"],
                            bootstrap = location_rf_options["bootstrap"],
                            oob_score = location_rf_options["oob_score"],
                            n_jobs = location_rf_options["n_jobs"],
                            random_state = 3308,
                            verbose = location_rf_options["verbose"],
                            warm_start = location_rf_options["warm_start"],
                            class_weight = location_rf_options["class_weight"],
                            ccp_alpha = location_rf_options["ccp_alpha"],
                            max_samples = location_rf_options["max_samples"],
                        )
        return random_forest
    
    def _compute_roc_auc_cv(self, index, random_forest, predictors, labels) :
        """Function that calculates AUROC, fpr and tpr for a given model on a dataset in crossvalidation"""
        
        prediction_probs = random_forest.predict_proba(predictors.iloc[index])[:,1]
        fpr, tpr , thresholds= roc_curve(labels.iloc[index], prediction_probs) 
        auc_score = auc(fpr, tpr)
        return  fpr, tpr ,auc_score

    def _compute_roc_auc(self,  random_forest, predictors, labels) :
        """Function that calculates AUROC, fpr and tpr for a given model on a datasetn"""
        prediction_probs = random_forest.predict_proba(predictors)[:,1]
        fpr, tpr , thresholds= roc_curve(labels, prediction_probs) 
        auc_score = auc(fpr, tpr)
        return  fpr, tpr ,auc_score

   

    def _compute_metrics_cv(self, index, random_forest, predictors, labels) :
        """Function that calculates F1, Accuarcy, MCC, Sensitivity and Specificity of a given model on a dataset during Crossvalidation"""
        predictions = random_forest.predict(predictors.iloc[index])
        tn, fp, fn, tp = confusion_matrix(labels.iloc[index], predictions).ravel()
        f1 = f1_score(labels.iloc[index], predictions)
        acc = accuracy_score(labels.iloc[index], predictions)
        mcc = matthews_corrcoef(labels.iloc[index], predictions)
        sensitivity = tp/(tp+fn)
        specificity = tn/(tn+fp)
        return f1, acc, mcc, sensitivity, specificity

    def _compute_metrics(self, random_forest, predictors, labels) :
        """Function that calculates F1, Accuarcy, MCC, Sensitivity and Specificity of a given model on a dataset"""
        predictions = random_forest.predict(predictors)
        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
        f1 = f1_score(labels, predictions)
        acc = accuracy_score(labels, predictions)
        mcc = matthews_corrcoef(labels, predictions)
        sensitivity = tp/(tp+fn)
        specificity = tn/(tn+fp)
        return f1, acc, mcc, sensitivity, specificity

    def tune_hyperparameters(self) :
        """Function used to find hyperparameters output is best parameters found into a file"""
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]
        
        # Number of features to consider at every split
        max_features = ['sqrt']
        
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(50, 400, num = 10)]
        max_depth.append(None)
        
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        
        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}
        
        # Databases for with a seperate model has to be trained
        databases = ["MIMICIV", "uka", "eICU"]

        # Filter that has been used on the training dataset 
        filters = ["full", "extreme", "light"]

        # Scorer to find bestperforming RF
        score = make_scorer(self.custom_scorer, greater_is_better=True)
        location_rf_options = self.random_forest_params
        
        # Train a models for every database for data with different filters a apllied (Results in best model for each combination of database and filter)
        for db in databases:
            for f in filters:
                name = db + "_data_" + f
                training_location  = "../Data/Training_Data/" + name + ".parquet"
                test_location = "../Data/Test_Data/" + name + ".parquet"
                
                # Load training and test data
                training_predictors, training_labels = self._read_data(training_location)
                test_predictors, test_labels = self._read_data(test_location)
                
                # Init RFs with hyperparameter grid and base hyperparameters
                rf = RandomForestClassifier()
                rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=15, cv=5, random_state=3308, n_jobs=-1, scoring=score)
                base = RandomForestClassifier(
                            n_estimators = location_rf_options["n_estimators"],
                            criterion = location_rf_options["criterion"],
                            max_depth = location_rf_options["max_depth"],
                            min_samples_split = location_rf_options["min_samples_split"],
                            min_samples_leaf = location_rf_options["min_samples_leaf"],
                            min_weight_fraction_leaf = location_rf_options["min_weight_fraction_leaf"],
                            max_features = location_rf_options["max_features"],
                            max_leaf_nodes = location_rf_options["max_leaf_nodes"],
                            min_impurity_decrease =  location_rf_options["min_impurity_decrease"],
                            bootstrap = location_rf_options["bootstrap"],
                            oob_score = location_rf_options["oob_score"],
                            n_jobs = location_rf_options["n_jobs"],
                            random_state = 3308,
                            verbose = location_rf_options["verbose"],
                            warm_start = location_rf_options["warm_start"],
                            class_weight = location_rf_options["class_weight"],
                            ccp_alpha = location_rf_options["ccp_alpha"],
                            max_samples = location_rf_options["max_samples"],
                        )
                
                # Find best hyperparmeters in grid
                rf_random = rf_random.fit(training_predictors, training_labels)
                best_random = rf_random.best_params_
                base = base.fit(training_predictors, training_labels)

                # Evaluate performance
                base_performance = self.evaluate_tuning(base, test_predictors, test_labels)
                random_performance = self.evaluate_tuning(rf_random, test_predictors, test_labels)
                print(name)
                print('Improvement of {:0.6f}%.'.format( 100 * (random_performance - base_performance) / base_performance))
                
                # Store best hyperparameters
                param_location = "../Data/Models/" + name + ".json"
                file = open(param_location, 'w')
                json.dump(best_random, file)
                file.close()


                

    # Evaluator using the MCC
    def evaluate_tuning(self, model, test_predictors, test_labels):
        """Used during hyperparameter tuning to estimate performance improvement"""
        _, _, mcc, _, _ = self._compute_metrics(model, test_predictors, test_labels)
        print("MCC: " + str(mcc))
        return mcc
    
    
    # Functions for loading and storing RF models
    def _store_model(self, random_forest) :
        model_path = "../Data/Models/" + self.learning_params_general["model_name"]
        file = open(model_path, "wb")
        pickle.dump(random_forest, file)
        file.close()

    def _load_model(self) :
        model_path = "../Data/Models/" + self.learning_params_general["model_name"]
        file = open(model_path, "rb")
        random_forest = pickle.load(file)
        file.close()
        return random_forest

    
