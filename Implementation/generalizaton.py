
from sklearn.model_selection import StratifiedKFold


from sklearn.feature_selection import  mutual_info_classif, SelectFromModel, SelectPercentile
from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, matthews_corrcoef, roc_curve, make_scorer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import json

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
    best_models = ["uka_data_extreme", "eICU", "MIMICIV-AC-RF"]
    datasets = ["uka_data_extreme", "eICU_data_extreme", "MIMICIV_data_extreme"]
    
    for model in datasets:
        location_config = "../Data/Models/Save/" + model + ".json"
        location_testing = []
        location_Metrics = []
        options = read_options(location_config)
        rf = RandomForestClassifier(
                                n_estimators = options["n_estimators"], min_samples_split = options["min_samples_split"], min_samples_leaf = options["min_samples_leaf"], 
                                max_features = "sqrt", max_depth = options["max_depth"], bootstrap = options["bootstrap"]
                            )
        if "uka" in model:
            location_training = "../Data/Training_Data/uka_data_extreme.parquet"
            location_testing.append("../Data/Test_Data/eICU_data_extreme.parquet")
            location_testing.append("../Data/Test_Data/uka_data_extreme.parquet")
            location_testing.append("../Data/Test_Data/MIMICIV_data_extreme.parquet")
            location_Metrics.append("../Data/Results/uka_eICU.json")
            location_Metrics.append("../Data/Results/uka_uka.json")
            location_Metrics.append("../Data/Results/uka_MIMICIV.json")


        elif "eICU" in model :
            location_training = "../Data/Training_Data/eICU_data_extreme.parquet"
            location_testing.append("../Data/Test_Data/eICU_data_extreme.parquet")
            location_testing.append("../Data/Test_Data/uka_data_extreme.parquet")
            location_testing.append("../Data/Test_Data/MIMICIV_data_extreme.parquet")
            location_Metrics.append("../Data/Results/eICU_eICU.json")
            location_Metrics.append("../Data/Results/eICU_uka.json")
            location_Metrics.append("../Data/Results/eICU_MIMICIV.json")
        elif "MIMICIV" in model :
            location_training = "../Data/Training_Data/MIMICIV_data_extreme.parquet"
            location_testing.append("../Data/Test_Data/eICU_data_extreme.parquet")
            location_testing.append("../Data/Test_Data/uka_data_extreme.parquet")
            location_testing.append("../Data/Test_Data/MIMICIV_data_extreme.parquet")
            location_Metrics.append("../Data/Results/MIMICIV_eICU.json")
            location_Metrics.append("../Data/Results/MIMICIV_uka.json")
            location_Metrics.append("../Data/Results/MIMICIV_MIMICIV.json")
        data = pd.read_parquet(location_training, engine='auto')
        data = data.reset_index()
        del data['index']
        labels = data["ARDS"]
        predictors = data.loc[:, data.columns != 'ARDS']
        clf = RandomForestClassifier(
                                n_estimators = options["n_estimators"], min_samples_split = options["min_samples_split"], min_samples_leaf = options["min_samples_leaf"], max_features = "sqrt", max_depth = options["max_depth"], bootstrap = options["bootstrap"]
                            )
        clf = clf.fit(predictors, labels)
        model = SelectFromModel(clf, prefit=True)
        cols = model.get_support(indices=True)
        fs_predictors = predictors.iloc[:, cols]
        rf = rf.fit(fs_predictors, labels)
        print("After learning")
        for index in range(len(location_testing)):
            predictors, labels = _read_data(location_testing[index])
            predictors = predictors.iloc[:,cols]
            evaluate(rf, predictors, labels, location_Metrics[index])


            

