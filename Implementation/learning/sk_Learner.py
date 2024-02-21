from typing import Any
import pandas as pd
import numpy as np

import pickle
import json

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

from learning.ILearner import ILearner
from utils import *

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, matthews_corrcoef, roc_curve, make_scorer

threshold_calcs = [
    ["geometric_root", lambda fpr, tpr: tpr * (1 - fpr)],
    ["max_tpr", lambda fpr, tpr: tpr],
    ["max_tpr_min_fpr", lambda fpr, tpr: tpr - fpr],
    ["standard", None]
]


class SKLearner(ILearner):
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

    def find_best_proba(self, threshold_calc, tpr, fpr, thresholds):
        calc_name = threshold_calc[0]
        calc_func = threshold_calc[1]
        if calc_name != "standard":

            optimal_idx = np.argmax(calc_func(fpr, tpr))
            optimal_threshold = thresholds[optimal_idx]
        else:
            optimal_threshold = 0.5
        return optimal_threshold

    def learn_modular(self, rf: RandomForestClassifier, predictors, label, metrics: Metrics, model_path):
        # Learn and store resulting model
        print("In learn")
        random_forest = rf.fit(predictors, label)
        fpr, tpr, thresholds, auc_score = self._compute_roc_auc(random_forest, predictors, label)

        for threshold_calc in threshold_calcs:
            optimal_threshold = self.find_best_proba(threshold_calc, tpr, fpr, thresholds)
            calc_name = threshold_calc[0]

            metrics_list = [GenericMetric(calc_name, "fpr_train", "training", fpr),
                            GenericMetric(calc_name, "tpr_train", "training", tpr),
                            GenericMetric(calc_name, "auc_scores_train", "training", auc_score),
                            GenericMetric(calc_name, "optimal_probability", "training", optimal_threshold)]
            metrics.update_metrics(metrics_list)
        self._store_model(random_forest, metrics, model_path=model_path)

    def _learn(self):
        """Function that starts the learning process of the RF and stores the resulting model after completion"""

        # Init forest and read training data
        random_forest = self._init_forest(self.random_forest_params)
        predictors, label = self._read_training_data()
        dataset_name = self._determine_dataset()
        metrics_location = "../Data/Models/" + self.learning_params_general["model_name"] + "_metrics_train.json"
        metrics_name = self.learning_params_general["model_name"] + "_metrics_train"
        metrics = Metrics(dataset_name, metrics_location, metrics_name)
        model_path = "../Data/Models/" + self.learning_params_general["model_name"] + ".pkl"
        self.learn_modular(random_forest, predictors, label, metrics, model_path)

        return

    def custom_scorer(self, labels, prediction_probas, **kwargs) -> float:

        thresholdcalc = kwargs.get("threshold_calc", None)
        if thresholdcalc is None:
            print("Threshold calc is empty")
            return 0.0

        # Calculate tpr, fpr and threshold
        fpr, tpr, thresholds = roc_curve(labels, prediction_probas)

        # Get function for treshhold calculation
        optimal_threshold = self.find_best_proba(thresholdcalc, tpr, fpr, thresholds)
        predictions = (prediction_probas[:] > optimal_threshold).astype(int)

        # Calculate mcc and return as score
        _, mcc, _, _, _ = self._compute_metrics(labels, predictions)
        return mcc

    def _determine_dataset(self):
        prefix = "training_"
        keys = [prefix + name for name in ["uka", "MIMICIV", "eICU"]]
        dataset_name = ""
        for key in keys:
            active = self.learning_params_general[key]
            dataset_name += key if active == 1 else ""
        dataset_name.replace(prefix, "")
        return dataset_name

    # TODO Rework for best probability DONE??
    def _cross_validate_forest(self):
        """Function that perfroms crossvalidation"""

        settings_name = self.learning_params_general["model_name"]
        settings_path = self.learning_locations["validation_options_path"] + settings_name + ".json"
        metrics_path = self.learning_locations["validation_metrics_path"]
        metrics_name = settings_name

        # Load settings for RF
        file = open(settings_path, "r")
        rf_settings = json.load(file)
        print(rf_settings)
        print(metrics_path)
        file.close()

        # Initialize forrest and read data
        random_forest = self._init_forest(rf_settings)
        predictors, labels = self._read_training_data()
        self.cross_validate_forest_modular(random_forest, predictors, labels, metrics_path, metrics_name)

    def cross_validate_forest_modular(self, random_forest, predictors, labels, metrics_path, metrics_name):
        dataset_name = self._determine_dataset()
        # Create Splits for Crossvalidation
        cross_validation = StratifiedKFold(n_splits=self.cross_validation_params["n_splits"],
                                           shuffle=False, random_state=None)
        cross_validation_sets = list(cross_validation.split(predictors, labels))

        # Variables for perfromance metrics

        overall_metrics = Metrics(dataset_name=dataset_name, metrics_location=metrics_path, metrics_name=metrics_name)
        # Perform crossvalidation
        for calc in threshold_calcs:
            calc_name = calc[0]
            calc_func = calc[1]
            for (train_set, test_set), i in zip(cross_validation_sets,
                                                range(self.cross_validation_params["n_splits"])):
                predictors_train = predictors.iloc[train_set]
                expected_labels_train = labels.iloc[train_set]
                predictors_test = predictors.iloc[test_set]
                expected_labels_test = labels.iloc[test_set]

                # Learn model for the split
                fitted_rf = random_forest.fit(predictors_train, expected_labels_train)

                # Evaluation of model
                print("Training")
                fpr_train, tpr_train, thresholds, auc_score_train = self._compute_roc_auc_cv(fitted_rf,
                                                                                             predictors_train,
                                                                                             expected_labels_train)
                optimal_threshold = self.find_best_proba(calc, tpr_train, fpr_train, thresholds)

                print("Eval")
                fpr_eval, tpr_eval, _, auc_score_eval = self._compute_roc_auc_cv(fitted_rf, predictors_test,
                                                                                 expected_labels_test)
                f1, acc, mcc, sensitivity, specificity = self._compute_metrics_cv(fitted_rf, predictors_test,
                                                                                  expected_labels_test,
                                                                                  optimal_threshold)

                metrics_list = []

                metrics_list.append(GenericMetric(calc_name, "fpr_train", i, fpr_train))
                metrics_list.append(GenericMetric(calc_name, "tpr_train", i, tpr_train))
                metrics_list.append(GenericMetric(calc_name, "auc_scores_train", i, auc_score_train))
                metrics_list.append(GenericMetric(calc_name, "fpr_eval", i, fpr_eval))
                metrics_list.append(GenericMetric(calc_name, "tpr_eval", i, tpr_eval))
                metrics_list.append(GenericMetric(calc_name, "auc_scores_eval", i, auc_score_eval))
                metrics_list.append(GenericMetric(calc_name, "sensitivity", i, sensitivity))
                metrics_list.append(GenericMetric(calc_name, "specificity", i, specificity))
                metrics_list.append(GenericMetric(calc_name, "accuracy", i, acc))
                metrics_list.append(GenericMetric(calc_name, "mcc", i, mcc))
                metrics_list.append(GenericMetric(calc_name, "f1", i, f1))
                metrics_list.append(GenericMetric(calc_name, "optimal_probability", i, optimal_threshold))

                overall_metrics.update_metrics(metrics_list)

        overall_metrics.save_metrics()

    def evaluate_modular(self, random_forest, predictors, labels, metrics_full: Metrics):
        # Compute evaluation results
        optimal_thresholds = metrics_full.optimal_probabilities
        fpr, tpr, thresholds, auc_score = self._compute_roc_auc(random_forest, predictors, labels)
        for calc_name, _ in optimal_thresholds.items():
            threshold = optimal_thresholds[calc_name]["training"]
            f1, acc, mcc, sensitivity, specificity = self._compute_metrics_eval(random_forest, predictors, labels,
                                                                                threshold)
            metrics_list = [
                GenericMetric(calc_name, "fpr_eval", "eval", fpr),
                GenericMetric(calc_name, "tpr_eval", "eval", tpr),
                GenericMetric(calc_name, "auc_scores_eval", "eval", auc_score),
                GenericMetric(calc_name, "accuracy", "eval", acc),
                GenericMetric(calc_name, "mcc", "eval", mcc),
                GenericMetric(calc_name, "f1", "eval", f1),
                GenericMetric(calc_name, "sensitivity", "eval", sensitivity),
                GenericMetric(calc_name, "specificity", "eval", specificity)
            ]
            metrics_full.update_metrics(metrics_list)
        metrics_full.save_metrics()

    def evaluate(self):
        """Function used to evalute a RF model"""

        # Load model and test data
        print("Evaluating")
        model_path = "../Data/Models/" + self.learning_params_general["model_name"]
        metrics_location = "../Data/Models/"
        metrics_name = self.learning_params_general["model_name"] + "_metrics_train"
        dataset_name = self._determine_dataset()
        random_forest, optimal_thresholds, metrics_full = self._load_model(model_path, metrics_location, metrics_name,
                                                                           dataset_name)
        predictors, labels = self._read_test_data()
        self.evaluate_modular(random_forest, optimal_thresholds, predictors, metrics_full)

        return

    def _init_forest(self, location_rf_options) -> RandomForestClassifier:
        """Function that intializes the Random Forest"""

        # Check if new split into Training and Validation set should be performed
        if self.learning_params_general["generate_new_sets"] == 1:
            self._generate_new_full_dataset()
        print(location_rf_options)
        # Init RF
        random_forest = RandomForestClassifier(
            n_estimators=location_rf_options["n_estimators"],
            criterion=location_rf_options["criterion"],
            max_depth=location_rf_options["max_depth"],
            min_samples_split=location_rf_options["min_samples_split"],
            min_samples_leaf=location_rf_options["min_samples_leaf"],
            min_weight_fraction_leaf=location_rf_options["min_weight_fraction_leaf"],
            max_features=location_rf_options["max_features"],
            max_leaf_nodes=location_rf_options["max_leaf_nodes"],
            min_impurity_decrease=location_rf_options["min_impurity_decrease"],
            bootstrap=location_rf_options["bootstrap"],
            oob_score=location_rf_options["oob_score"],
            n_jobs=location_rf_options["n_jobs"],
            random_state=3308,
            verbose=location_rf_options["verbose"],
            warm_start=location_rf_options["warm_start"],
            class_weight=location_rf_options["class_weight"],
            ccp_alpha=location_rf_options["ccp_alpha"],
            max_samples=location_rf_options["max_samples"],
        )
        return random_forest

    def _compute_roc_auc_cv(self, random_forest, predictors, labels):
        """Function that calculates AUROC, fpr and tpr for a given model on a dataset in crossvalidation"""

        prediction_probs = random_forest.predict_proba(predictors)[:, 1]
        print("Compute ROC AUC")

        fpr, tpr, thresholds = roc_curve(labels, prediction_probs)

        auc_score = auc(fpr, tpr)
        return fpr, tpr, thresholds, auc_score

    @staticmethod
    def _compute_roc_auc(random_forest: RandomForestClassifier, predictors, labels):
        """Function that calculates AUROC, fpr and tpr for a given model on a datasetn"""
        prediction_probs = random_forest.predict_proba(predictors)[:, 1]
        fpr, tpr, thresholds = roc_curve(labels, prediction_probs)
        auc_score = auc(fpr, tpr)
        return fpr, tpr, thresholds, auc_score

    @staticmethod
    def _compute_metrics(expected_values, predicted_values):
        tn, fp, fn, tp = confusion_matrix(expected_values, predicted_values).ravel()
        f1 = f1_score(expected_values, predicted_values)
        acc = accuracy_score(expected_values, predicted_values)
        mcc = matthews_corrcoef(expected_values, predicted_values)
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        return f1, acc, mcc, sensitivity, specificity

    def _compute_metrics_cv(self, random_forest, predictors, labels, threshold_proba):
        """Function that calculates F1, Accuarcy, MCC, Sensitivity and Specificity of a given model on a dataset
        during Crossvalidation"""
        prediction_probas = random_forest.predict_proba(predictors)

        predictions = (prediction_probas[:, 1] > threshold_proba).astype(int)
        expected_values = labels

        f1, acc, mcc, sensitivity, specificity = self._compute_metrics(expected_values, predictions)

        return f1, acc, mcc, sensitivity, specificity

    def _compute_metrics_eval(self, random_forest, predictors, expected_values, threshold_proba):
        """Function that calculates F1, Accuarcy, MCC, Sensitivity and Specificity of a given model on a dataset"""
        prediction_probas = random_forest.predict_proba(predictors)

        predictions = (prediction_probas[:, 1] > threshold_proba).astype(int)
        f1, acc, mcc, sensitivity, specificity = self._compute_metrics(expected_values, predictions)

        return f1, acc, mcc, sensitivity, specificity

    # TODO think of a way to do tuning with tresholdcalc
    def tune_hyperparameters(self, db: str, filtering: str, location_rf_options=None):
        """Function used to find hyperparameters output is best parameters found into a file"""
        if location_rf_options is None:
            location_rf_options = self.random_forest_params
            print("Using options")

        print(db, filtering)
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start=100, stop=1000, num=10)]

        # Splitting criterion
        criterion = ['gini', 'entropy', 'log_loss']

        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(50, 400, num=10)]
        max_depth.append(None)

        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]

        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]

        #
        min_weight_fraction_leaf = [0.0]

        # Number of features to consider at every split
        max_features = ['sqrt']

        max_leaf_nodes = [None]

        min_impurity_decrease = [0.0]

        # Method of selecting samples for training each tree
        bootstrap = [True]

        oob_score = [False]

        n_jobs = [-1]

        random_state = [3308]

        verbose = [0]

        warm_start = [False]

        class_weight = [None]

        ccp_alpha = [0.0]

        max_samples = [None]

        # Create the random grid
        grid = {'n_estimators': n_estimators,
                'criterion': criterion,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'min_weight_fraction_leaf': min_weight_fraction_leaf,
                'max_features': max_features,
                'max_leaf_nodes': max_leaf_nodes,
                'min_impurity_decrease': min_impurity_decrease,
                'bootstrap': bootstrap,
                'oob_score': oob_score,
                'n_jobs': n_jobs,
                'random_state': random_state,
                'verbose': verbose,
                'warm_start': warm_start,
                'class_weight': class_weight,
                'ccp_alpha': ccp_alpha,
                'max_samples': max_samples}

        # Scorer to find bestperforming RF
        for threshold_calc in threshold_calcs:
            score = make_scorer(self.custom_scorer, needs_proba=True, greater_is_better=True,
                                threshold_calc=threshold_calc)

            # Train a models for every database for data with different filters a apllied (Results in best model for each combination of database and filter)

            # Set paths for training and test data
            name = db + "_data_" + filtering
            print(name)

            training_location = "../Data/Training_Data/" + name + ".parquet"
            test_location = "../Data/Test_Data/" + name + ".parquet"
            model_name = name + "_" + threshold_calc[0]

            # Load training and test data
            training_predictors, training_labels = self._read_data(training_location)
            test_predictors, test_labels = self._read_data(test_location)
            # Init RFs with hyperparameter grid and base hyperparameters
            rf = RandomForestClassifier()
            rf_random = RandomizedSearchCV(estimator=rf, param_distributions=grid, n_iter=60, cv=5,
                                           random_state=3308, n_jobs=-1, scoring=score)
            # rf_random = GridSearchCV(estimator=rf, param_grid=grid, scoring=score, cv=5, n_jobs=-1)
            base = RandomForestClassifier(
                n_estimators=location_rf_options["n_estimators"],
                criterion=location_rf_options["criterion"],
                max_depth=location_rf_options["max_depth"],
                min_samples_split=location_rf_options["min_samples_split"],
                min_samples_leaf=location_rf_options["min_samples_leaf"],
                min_weight_fraction_leaf=location_rf_options["min_weight_fraction_leaf"],
                max_features=location_rf_options["max_features"],
                max_leaf_nodes=location_rf_options["max_leaf_nodes"],
                min_impurity_decrease=location_rf_options["min_impurity_decrease"],
                bootstrap=location_rf_options["bootstrap"],
                oob_score=location_rf_options["oob_score"],
                n_jobs=-1,
                random_state=3308,
                verbose=location_rf_options["verbose"],
                warm_start=location_rf_options["warm_start"],
                class_weight=location_rf_options["class_weight"],
                ccp_alpha=location_rf_options["ccp_alpha"],
                max_samples=location_rf_options["max_samples"]
            )
            # Find best hyperparmeters in grid
            rf_random = rf_random.fit(training_predictors, training_labels)
            random_fpr, random_tpr, random_thresholds, _ = self._compute_roc_auc(rf_random, training_predictors,
                                                                                 training_labels)
            optimal_threshold_random = self.find_best_proba(threshold_calc, random_tpr, random_fpr, random_thresholds)
            best_random = rf_random.best_params_
            base = base.fit(training_predictors, training_labels)
            base_performance = self.evaluate_tuning(base, test_predictors, test_labels, optimal_threshold_random)

            base_fpr, base_tpr, base_thresholds, _ = self._compute_roc_auc(base, training_predictors, training_labels)
            optimal_threshold_base = self.find_best_proba(threshold_calc, base_tpr, base_fpr, base_thresholds)
            # Evaluate performance

            random_performance = self.evaluate_tuning(rf_random, test_predictors, test_labels, optimal_threshold_base)
            print(model_name)
            print(
                'Improvement of {:0.6f}%.'.format(
                    100 * (random_performance - base_performance) / base_performance))
            # Store best hyperparameters
            param_location = "../Data/Models/" + model_name + ".json"

            file = open(param_location, 'w')
            json.dump(best_random, file)
            file.close()

    # Evaluator using the MCC
    def evaluate_tuning(self, model, test_predictors, test_labels, threshold):
        """Used during hyperparameter tuning to estimate performance improvement"""

        _, _, mcc, _, _ = self._compute_metrics_eval(model, test_predictors, test_labels, threshold)
        print("MCC: " + str(mcc))
        return mcc

    # Functions for loading and storing RF models
    def _store_model(self, random_forest, metrics_training: Metrics, model_path):


        file = open(model_path, "wb")
        pickle.dump(random_forest, file)
        file.close()
        metrics_training.save_metrics()

    def _load_model(self, model_path, metrics_location, metrics_name, dataset_name):

        metrics_path = metrics_location + metrics_name + ".json"
        file = open(model_path, "rb")
        random_forest = pickle.load(file)
        file.close()
        file = open(metrics_path, "r")
        metrics_training_json = json.load(file)
        file.close()

        metrics_training = Metrics(dataset_name, metrics_location, metrics_name)
        metrics_training.fprs_train = metrics_training_json["fprs_train"]
        metrics_training.tprs_train = metrics_training_json["tprs_train"]
        metrics_training.auc_scores_training = metrics_training_json["auc_scores_train"]
        metrics_training.optimal_probabilities = metrics_training_json["optimal_probabilities"]
        optimal_thresholds = metrics_training_json["optimal_probabilities"]
        return random_forest, optimal_thresholds, metrics_training
