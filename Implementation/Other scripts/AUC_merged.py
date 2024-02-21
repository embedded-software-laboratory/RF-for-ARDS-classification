"""Script to print multiple ROC curves to pdf"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import auc

plt.rcParams.update({'font.size': 20})
def read_options(path):
    with open(path, "r") as file:
        return json.load(file)


def plot_roc_curve(fprs, tprs, f, ax, label, color, linestyle):
    """Plot the Receiver Operating Characteristic from a list
    of true positive rates and false positive rates."""
    # Initialize useful lists + the plot axes.
    tprs_interp = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    print(fprs)
    # Compute mean ROC + AUC scores.
    for i, (fpr, tpr) in enumerate(zip(fprs, tprs)):
        print(fpr)
        print(tpr)
        print(mean_fpr)
        tprs_interp.append(np.interp(mean_fpr, fpr, tpr))
        tprs_interp[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

    # Plot the mean ROC.
    mean_tpr = np.mean(tprs_interp, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color=color,
            label=r'%s (AUC = %0.2f $\pm$ %0.2f)' % (label, mean_auc, std_auc),
            linestyle=linestyle,
            lw=2, alpha=.8)

    # Plot the standard deviation around the mean ROC.
    std_tpr = np.std(tprs_interp, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2)

    return (f, ax, mean_auc)


if __name__ == '__main__':
    datasets = ["eICU", "MIMICIV", "uka"]
    revision = 7
    for dataset in datasets:
        for fs in ["30", "60", "RF", None]:
            f, ax = plt.subplots(figsize=(14, 10))

            print(f"Plotting {dataset} data with {fs} features [revision {revision}]...")

            for filter in ["full", "light", "extreme"]:
                path = f"../../Data/Validation/Metrics/v7/{dataset}_data_{filter}_{fs}_Version_{revision}.json"
                results = read_options(path)
                color = "g" if filter == "full" else "b" if filter == "light" else "r"
                label = "No filter" if filter == "full" else "BD + Lite" if filter == "light" else "BD + Strict"
                linestyle = ":" if filter == "full" else "-." if filter == "light" else "--"
                tpr_list = [ results["tprs_eval"]["geometric_root"][str(i)] for i in
                             range(0, len(results["tprs_eval"]["geometric_root"])-1)]
                fpr_list = [results["fprs_eval"]["geometric_root"][str(i)] for i in
                            range(0, len(results["fprs_eval"]["geometric_root"])-1)]
                roc = plot_roc_curve(fpr_list, tpr_list, f, ax, label, color, linestyle)

            # Plot the luck line.
            ax.plot([0, 1], [0, 1], linestyle='-', lw=2, color='grey', label='Coin toss', alpha=.8)

            patches, _ = ax.get_legend_handles_labels()
            patches.append(mpatches.Patch(color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.'))

            ax.legend(loc="lower right", handles=patches)
            ax.set_xlim([-0.05, 1.05])
            ax.set_ylim([-0.05, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Receiver operating characteristic curve')

            outname = f"../../Data/Validation/Metrics/v7/{dataset}_data_{fs}_Version_{revision}.pdf"
            plt.savefig(outname)
