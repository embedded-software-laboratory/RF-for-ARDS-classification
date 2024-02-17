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
        print(mean_fpr)
        print(fpr)
        print(tpr)
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

    plt.savefig(location, format="pdf")
    return mean_auc

if __name__ == "__main__" :
    databases = [ "eICU", "uka", "MIMICIV"]
    filters = ["full", "light", "extreme"]
    feature = ["None", "RF", "30", "60"]
    revision = 4
    metrics = []
    

    for db in databases:
        metrics_db = []
        for f in filters:
            name = db + "_data_" + f + str(revision) + ".parquet"
            location = "../../Data/Validation/Metrics/Rev4/"
            location_write = location+name
            metrics_filter = []
            print(f)
            for fs in feature :
                name_read = db + "_data_" + f + "_" + fs + str(revision)
                location_read = location + name_read + ".json"

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
                name_metric = db + "-" + fn + "_" + fsn
                print(name_metric)
                for threshold in ["standard", "max_tpr", "geometric_root"]:
                
                    auc_location =  location + "ROC/" + name_metric + "_" + threshold + ".pdf"
                    tprs = data_json["tprs_eval"][threshold]
                    del tprs["mean"]
                    tprs_list = []
                    for key, value in tprs.items():
                        tprs_list.append(value)
                    fprs = data_json["fprs_eval"][threshold]
                    del fprs["mean"]
                    fprs_list = []
                    for key, value in fprs.items():
                        fprs_list.append(value)
                    plot_roc_curve(fprs_list, tprs_list, 0, auc_location)


            
        