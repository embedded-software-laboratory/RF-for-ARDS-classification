from sklearn.model_selection import StratifiedKFold

from sklearn.feature_selection import mutual_info_classif, SelectFromModel, SelectPercentile
from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, matthews_corrcoef, roc_curve, make_scorer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import json
import argparse
from learning.sk_Learner import SKLearner

def read_options(location):
    with open(location, "r") as file:
        return json.load(file)


if __name__ == "__main__":
    type_of_learn = "Single"
    revision_num = 1
    revision = type_of_learn + "_" + str(revision_num)
    # databases  = ["eICU", "uka", "MIMICIV"]
    argparser = argparse.ArgumentParser()
    argparser.add_argument("database")
    args = argparser.parse_args()
    databases = [args.database]
    filters = ["full", "light", "extreme"]
    feature_selections = ["None", "30", "60", "RF"]


    for db in databases:
        for fil in filters:
            name = db + "_data_" + fil

            location_training = "../Data/Training_Data/" + name + ".parquet"
            location_config = "../Data/Models/Save/Rev1/" + name + ".json"
            data = pd.read_parquet(location_training, engine='auto')

            data = data.reset_index()
            del data['index']
            labels = data["ARDS"]
            predictors = data.loc[:, data.columns != 'ARDS']
            options = read_options(location_config)
            options_general = read_options("./options.json")
            rf = RandomForestClassifier(
                n_estimators=options["n_estimators"], min_samples_split=options["min_samples_split"],
                min_samples_leaf=options["min_samples_leaf"], max_features="sqrt", max_depth=options["max_depth"],
                bootstrap=options["bootstrap"]
            )
            for fs in feature_selections:
                fs_predictors = None
                match fs:
                    case "30":
                        model = SelectPercentile(mutual_info_classif, percentile=30)
                        model.fit(predictors, labels)
                        cols = model.get_support(indices=True)
                        fs_predictors = predictors.iloc[:, cols]
                    case "60":
                        model = SelectPercentile(mutual_info_classif, percentile=60)
                        model.fit(predictors, labels)
                        cols = model.get_support(indices=True)
                        fs_predictors = predictors.iloc[:, cols]
                    case "RF":
                        clf = RandomForestClassifier(
                            n_estimators=options["n_estimators"], min_samples_split=options["min_samples_split"],
                            min_samples_leaf=options["min_samples_leaf"], max_features="sqrt",
                            max_depth=options["max_depth"], bootstrap=options["bootstrap"]
                        )
                        clf = clf.fit(predictors, labels)
                        model = SelectFromModel(clf, prefit=True)
                        cols = model.get_support(indices=True)
                        fs_predictors = predictors.iloc[:, cols]
                    case "None":
                        fs_predictors = predictors
                name_model = db + "_data_" + fil + "_" + fs
                location_features = "../Data/Validation/Settings/" + name_model + revision + "_features.csv"
                location_metrics = "../Data/Validation/Metrics/"
                name_metrics = name_model + revision
                print(name_model)
                learner = SKLearner(options_general)
                learner.learn_modular(rf, fs_predictors, labels, "Testing", location_metrics, name_metrics)

                selected_features = fs_predictors.columns.to_series()
                selected_features.to_csv(location_features, sep=",", index=False)
