
from sklearn.model_selection import StratifiedKFold


from sklearn.feature_selection import  mutual_info_classif, SelectFromModel, SelectPercentile
from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, matthews_corrcoef, roc_curve, make_scorer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import json

def read_options(location):
    with open(location, "r") as file:
        return json.load(file)


def cross_validate_forest(rf, metrics_path, predictors, labels) :
        

       
        print(metrics_path)

        

        cross_validation = StratifiedKFold(n_splits=5, shuffle=False, random_state=None)
    
        fprs, tprs, scores, sensitivities, specificities, accuracies, mccs, f1s = [], [], [], [],[], [], [], []
        
        for (train_set, test_set), i in zip(cross_validation.split(predictors, labels), range(0,5,1)):
            predictors_train = predictors.iloc[train_set]
            labels_train = labels.iloc[train_set]
            
            fitted = rf.fit(predictors_train, labels_train)
            _, _, auc_score_training = _compute_roc_auc_cv(train_set, fitted, predictors, labels) 
            fpr, tpr, auc_score_test = _compute_roc_auc_cv(test_set, fitted, predictors, labels) 
            f1, acc, mcc, sensitivity, specificity = _compute_metrics_cv(test_set, fitted, predictors, labels)
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
        print("metrics_path")
        file = open(metrics_path, 'w')
        json.dump(metric_dict, file)
        file.close()
            
def _compute_roc_auc_cv( index, random_forest, predictors, labels) :
        prediction_probs = random_forest.predict_proba(predictors.iloc[index])[:,1]
        fpr, tpr , thresholds= roc_curve(labels.iloc[index], prediction_probs) 
        auc_score = auc(fpr, tpr)
        return  fpr, tpr ,auc_score

def _compute_metrics_cv(index, random_forest, predictors, labels) :
        predictions = random_forest.predict(predictors.iloc[index])
        tn, fp, fn, tp = confusion_matrix(labels.iloc[index], predictions).ravel()
        f1 = f1_score(labels.iloc[index], predictions)
        acc = accuracy_score(labels.iloc[index], predictions)
        mcc = matthews_corrcoef(labels.iloc[index], predictions)
        sensitivity = tp/(tp+fn)
        specificity = tn/(tn+fp)
        return f1, acc, mcc, sensitivity, specificity

if __name__ == "__main__":
    revision = 2
    databases  = ["eICU", "uka", "MIMICIV"]
    filters = ["full", "light", "extreme"]
    feature_selections = ["None", "30", "60", "RF"]

    for db in databases:
        for fil in filters:
            name = db + "_data_" + fil 

            location_training = "../Data/Training_Data/" + name + ".parquet"
            location_config = "../Data/Models/Save/" + name + ".json"
            data = pd.read_parquet(location_training, engine='auto')
            data = data.reset_index()
            del data['index']
            labels = data["ARDS"]
            predictors = data.loc[:, data.columns != 'ARDS']
            print(data.shape)
            options = read_options(location_config)
            rf = RandomForestClassifier(
                                n_estimators = options["n_estimators"], min_samples_split = options["min_samples_split"], min_samples_leaf = options["min_samples_leaf"], max_features = "sqrt", max_depth = options["max_depth"], bootstrap = options["bootstrap"]
                            )
            for fs in feature_selections:
                fs_predictors = None
                match fs:
                    case "30" :
                        model = SelectPercentile(mutual_info_classif, percentile = 30)
                        model.fit(predictors, labels)
                        cols = model.get_support(indices=True)
                        fs_predictors = predictors.iloc[:, cols]
                    case "60" :
                        model = SelectPercentile(mutual_info_classif, percentile = 60)
                        model.fit(predictors, labels)
                        cols = model.get_support(indices=True)
                        fs_predictors = predictors.iloc[:, cols]
                    case "RF" :
                        clf = RandomForestClassifier(
                                n_estimators = options["n_estimators"], min_samples_split = options["min_samples_split"], min_samples_leaf = options["min_samples_leaf"], max_features = "sqrt", max_depth = options["max_depth"], bootstrap = options["bootstrap"]
                            )
                        clf = clf.fit(predictors, labels)
                        model = SelectFromModel(clf, prefit=True)
                        cols = model.get_support(indices=True)
                        fs_predictors = predictors.iloc[:, cols]
                    case "None" :
                        fs_predictors = predictors
                name_model = db + "_data_" + fil + fs            
                location_features = "../Data/Validation/Settings/" + name_model + str(revision) + ".csv"
                location_metrics = "../Data/Validation/Metrics/" + name_model + str(revision) +  ".json"


                cross_validate_forest(rf, location_metrics, fs_predictors, labels)
                selected_features = fs_predictors.columns.to_series()
                selected_features.to_csv(location_features, sep=",", index=False)

