
from sklearn.model_selection import StratifiedKFold


from sklearn.feature_selection import  mutual_info_classif, SelectFromModel, SelectPercentile
from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, matthews_corrcoef, roc_curve, make_scorer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from learning.sk_Learner import  SKLearner
import json

from utils import Metrics


def read_options(location):
    with open(location, "r") as file:
        return json.load(file)

def _read_data( location) -> pd.DataFrame:
        data = pd.read_parquet(location, engine='auto')
        data = data.reset_index()
        del data['index']
        label = data["ARDS"]
        predictors = data.loc[:, data.columns != 'ARDS']
        return predictors, label



def _compute_roc_auc(random_forest, predictors, labels) :
        prediction_probs = random_forest.predict_proba(predictors)[:,1]
        fpr, tpr , thresholds= roc_curve(labels, prediction_probs) 
        auc_score = auc(fpr, tpr)
        return  fpr, tpr ,auc_score


def _compute_metrics(random_forest, predictors, labels) :
        predictions = random_forest.predict(predictors)
        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
        f1 = f1_score(labels, predictions)
        acc = accuracy_score(labels, predictions)
        mcc = matthews_corrcoef(labels, predictions)
        sensitivity = tp/(tp+fn)
        specificity = tn/(tn+fp)
        return f1, acc, mcc, sensitivity, specificity

def evaluate(random_forest, predictors, labels, metrics_path) :
        
        fpr, tpr, auc_score = _compute_roc_auc(random_forest, predictors, labels) 
        f1, acc, mcc, sensitivity, specificity = _compute_metrics(random_forest, predictors, labels)

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

if __name__ == "__main__":
    best_models = ["uka_data_extreme_", "eICU", "MIMICIV-AC-RF"]
    datasets = ["uka_data_extreme", "eICU_data_extreme", "MIMICIV_data_extreme"]
    options_general = read_options("./options.json")
    
    for model_name in datasets:
        location_config = "../Data/Models/" + model_name + "_standard.json"
        location_testing = []
        path_metrics = "../Data/Results/"
        name_metrics = []
        dataset_name = []
        options = read_options(location_config)
        rf = RandomForestClassifier(
                n_estimators=options["n_estimators"],
                criterion=options["criterion"],
                max_depth=options["max_depth"],
                min_samples_split=options["min_samples_split"],
                min_samples_leaf=options["min_samples_leaf"],
                min_weight_fraction_leaf=options["min_weight_fraction_leaf"],
                max_features=options["max_features"],
                max_leaf_nodes=options["max_leaf_nodes"],
                min_impurity_decrease=options["min_impurity_decrease"],
                bootstrap=options["bootstrap"],
                oob_score=options["oob_score"],
                n_jobs=-1,
                random_state=3308,
                verbose=options["verbose"],
                warm_start=options["warm_start"],
                class_weight=options["class_weight"],
                ccp_alpha=options["ccp_alpha"],
                max_samples=options["max_samples"]
                            )
        if "uka" in model_name:
            location_training = "../Data/Training_Data/uka_data_extreme.parquet"
            location_testing.append("../Data/Test_Data/eICU_data_extreme.parquet")
            location_testing.append("../Data/Test_Data/uka_data_extreme.parquet")
            location_testing.append("../Data/Test_Data/MIMICIV_data_extreme.parquet")
            name_metrics.append("uka_eICU")
            name_metrics.append("uka_uka")
            name_metrics.append("uka_MIMICIV")
            model_path = "../Data/Models/uka_generalisation.pkl"


        elif "eICU" in model_name :
            location_training = "../Data/Training_Data/eICU_data_extreme.parquet"
            location_testing.append("../Data/Test_Data/eICU_data_extreme.parquet")
            location_testing.append("../Data/Test_Data/uka_data_extreme.parquet")
            location_testing.append("../Data/Test_Data/MIMICIV_data_extreme.parquet")
            name_metrics.append("eICU_eICU")
            name_metrics.append("eICU_uka")
            name_metrics.append("eICU_MIMICIV")
            model_path = "../Data/Models/eICU_generalisation.pkl"
        elif "MIMICIV" in model_name :
            location_training = "../Data/Training_Data/MIMICIV_data_extreme.parquet"
            location_testing.append("../Data/Test_Data/eICU_data_extreme.parquet")
            location_testing.append("../Data/Test_Data/uka_data_extreme.parquet")
            location_testing.append("../Data/Test_Data/MIMICIV_data_extreme.parquet")
            name_metrics.append("MIMICIV_eICU")
            name_metrics.append("MIMICIV_uka")
            name_metrics.append("MIMICIV_MIMICIV")
            model_path = "../Data/Models/MIMICIV_generalisation.pkl"
        data = pd.read_parquet(location_training, engine='auto')
        data = data.reset_index()
        del data['index']
        labels_learn = data["ARDS"]
        predictors = data.loc[:, data.columns != 'ARDS']
        clf = RandomForestClassifier(
                                n_estimators = options["n_estimators"], min_samples_split = options["min_samples_split"], min_samples_leaf = options["min_samples_leaf"], max_features = "sqrt", max_depth = options["max_depth"], bootstrap = options["bootstrap"]
                            )
        clf = clf.fit(predictors, labels_learn)
        model = SelectFromModel(clf, prefit=True)
        cols = model.get_support(indices=True)
        fs_predictors = predictors.iloc[:, cols]
        learner = SKLearner(options_general)


        print("After learning")
        for index in range(len(location_testing)):
            print(name_metrics)
            print(location_testing[index])
            metrics = Metrics(dataset_name=model_name, metrics_location=path_metrics, metrics_name=name_metrics[index])
            learner.learn_modular(rf=rf, predictors=fs_predictors, label=labels_learn, metrics=metrics, model_path=model_path)
            predictors_train, labels_train = _read_data(location_testing[index])
            predictors_train = predictors_train.iloc[:,cols]
            learner.evaluate_modular(random_forest=rf, predictors=predictors_train, labels=labels_train, metrics_full=metrics)



            

