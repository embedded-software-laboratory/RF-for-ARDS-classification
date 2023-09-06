import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import roc_curve, auc
matplotlib.rcParams.update({'font.size': 20})


"""Script that finds best performing model based on MCC"""
def read_options(path):
    with open(path, "r") as file:
        return json.load(file)

def plot_roc_curve(fprs, tprs, best, location):
    """Plot the Receiver Operating Characteristic from a list
    of true positive rates and false positive rates."""
    # Initialize useful lists + the plot axes.
    tprs_interp = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    f, ax = plt.subplots(figsize=(14,10))
    
    # Plot ROC for each K-Fold + compute AUC scores.
    for i, (fpr, tpr) in enumerate(zip(fprs, tprs)):
        tprs_interp.append(np.interp(mean_fpr, fpr, tpr))
        tprs_interp[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        ax.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        
    # Plot the luck line.
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Coin toss', alpha=.8)
    
    # Plot the mean ROC.
    mean_tpr = np.mean(tprs_interp, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)
    
    # Plot the standard deviation around the mean ROC.
    std_tpr = np.std(tprs_interp, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')
    
    # Fine tune and show the plot.
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic')
    ax.legend(loc="lower right")
    if best == 1:
        #plt.show()
        plt.savefig(location, format="pdf")
    return mean_auc

if __name__ == "__main__" :
    databases = [ "eICU", "uka", "MIMICIV"]
    filters = ["full", "light", "extreme"]
    feature = ["None", "RF", "30", "60"]
    revision = 2
    metrics = []
    

    for db in databases:
        metrics_db = []
        for f in filters:
            name = db + "_data_" + f + str(revision) + ".parquet"
            location = "../../Data/Validation/Metrics/"
            location_write = location+name
            metrics_filter = []
            print(f)
            for fs in feature :
                name_read = db + "_data_" + f + fs +str(revision)
                location_read = location + name_read + ".json"
                data = pd.read_json(location_read).drop([1,2,3,4])[["mcc", "sens", "spec"]]
                data_json = read_options(location_read)
                if fs == "None":
                    fsn = "full"
                else :
                    fsn = fs
                if f == "full" :
                    fn = "None"
                elif f == "light" :
                    fn = "BC"
                elif f == "extreme":
                    fn = "AC"
                name_metric = db + "-" + fn + "-" + fsn 
                print(name_metric)
                data["Name"] = [name_metric]
                
                auc_location = "./" + name_metric + ".pdf"
                data["AUC"] = [plot_roc_curve(data_json["fprs"], data_json["tprs"], 1, auc_location)]
                metrics_filter.append(data[["Name","mcc", "sens", "spec", "AUC"]])
                metrics.append(data[["Name","mcc", "sens", "spec", "AUC"]])
            metric_filter = pd.concat(metrics_filter)
            metric_filter.sort_values(by=["mcc", "sens", "spec"],  inplace=True, ascending=False)
            metric_filter.reset_index(inplace=True)
            del metric_filter['index']
            
            best = metric_filter.iloc[0, :].to_frame().transpose()
            metrics_db.append(best)
        metric_db = pd.concat(metrics_db)
        metric_db.reset_index(inplace=True)
        del metric_db['index']  
        loc_db = "./" + db + ".csv"
        metric_db.to_csv(loc_db, sep=",", index=False, float_format='%.4f')
    metric = pd.concat(metrics)
    metric.reset_index(inplace=True)
    
    del metric['index']
    metric.to_csv("./Metrics.csv", sep=",", index=False, float_format='%.4f')

            
        