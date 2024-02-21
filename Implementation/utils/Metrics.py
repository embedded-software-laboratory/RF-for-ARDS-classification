import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from .Generic_Metric import GenericMetric
from sklearn.metrics import auc

matplotlib.rcParams.update({'font.size': 20})


class Metrics:
    def __init__(self, dataset_name: str, metrics_location: str = None, metrics_name: str = None,
                 dataset_number: int = None, tprs_train: dict = None,
                 fprs_train: dict = None, tprs_eval: dict = None,
                 fprs_eval: dict = None,
                 auc_scores_training: dict = None, auc_scores_eval: dict = None,
                 optimal_probabilities: dict = None, acc: dict = None, sens: dict = None,
                 spec: dict = None, f1: dict = None, mcc: dict = None):
        self._dataset_name = dataset_name
        self._dataset_number = dataset_number
        self._metrics_location = metrics_location
        self._metrics_name = metrics_name
        self._tprs_train = tprs_train if tprs_train is not None else {}
        self._fprs_train = fprs_train if fprs_train is not None else {}
        self._tprs_eval = tprs_eval if tprs_eval is not None else {}
        self._fprs_eval = fprs_eval if fprs_eval is not None else {}
        self._auc_scores_training = auc_scores_training if auc_scores_training is not None else {}
        self._auc_scores_eval = auc_scores_eval if auc_scores_eval is not None else {}
        self._optimal_probabilities = optimal_probabilities if optimal_probabilities is not None else {}
        self._acc = acc if acc is not None else {}
        self._sens = sens if sens is not None else {}
        self._spec = spec if spec is not None else {}
        self._f1 = f1 if f1 is not None else {}
        self._mcc = mcc if mcc is not None else {}

    @property
    def dataset_name(self):
        return self._dataset_name

    @dataset_name.setter
    def dataset_name(self, value):
        self._dataset_name = value

    @property
    def dataset_number(self):
        return self._dataset_number

    @dataset_number.setter
    def dataset_number(self, value):
        self._dataset_number = value

    @property
    def metrics_location(self):
        return self._metrics_location

    @metrics_location.setter
    def metrics_location(self, value):
        self._metrics_location = value

    @property
    def metrics_name(self):
        return self._metrics_name

    @metrics_name.setter
    def metrics_name(self, value):
        self._metrics_name = value

    @property
    def tprs_train(self):
        return self._tprs_train

    @tprs_train.setter
    def tprs_train(self, value):
        self._tprs_train = value

    @property
    def fprs_train(self):
        return self._fprs_train

    @fprs_train.setter
    def fprs_train(self, value):
        self._fprs_train = value

    @property
    def tprs_eval(self):
        return self._tprs_eval

    @tprs_eval.setter
    def tprs_eval(self, value):
        self._tprs_eval = value

    @property
    def fprs_eval(self):
        return self._fprs_eval

    @fprs_eval.setter
    def fprs_eval(self, value):
        self._fprs_eval = value

    @property
    def auc_scores_training(self):
        return self._auc_scores_training

    @auc_scores_training.setter
    def auc_scores_training(self, value):
        self._auc_scores_training = value

    @property
    def auc_scores_eval(self):
        return self._auc_scores_eval

    @auc_scores_eval.setter
    def auc_scores_eval(self, value):
        self._auc_scores_eval = value

    @property
    def optimal_probabilities(self):
        return self._optimal_probabilities

    @optimal_probabilities.setter
    def optimal_probabilities(self, value):
        self._optimal_probabilities = value

    @property
    def acc(self):
        return self._acc

    @acc.setter
    def acc(self, value):
        self._acc = value

    @property
    def sens(self):
        return self._sens

    @sens.setter
    def sens(self, value):
        self._sens = value

    @property
    def spec(self):
        return self._spec

    @spec.setter
    def spec(self, value):
        self._spec = value

    @property
    def f1(self):
        return self._f1

    @f1.setter
    def f1(self, value):
        self._f1 = value

    @property
    def mcc(self):
        return self._mcc

    @mcc.setter
    def mcc(self, value):
        self._mcc = value

    def _calculate_mean(self, dict_to_use: dict):
        dict_new = dict_to_use.copy()
        all_values = []
        if "mean" in dict_new:
            dict_new.pop("mean")
        for key, value in dict_new.items():
            # TODO correct mean calculation for lists
            if isinstance(value, list):
                all_values += value
            else:
                all_values.append(value)
        return sum(all_values) / len(all_values)

    def _set_params(self, param_name: str, key_threshold_calc: str, value):

        match param_name:

            case "optimal_probability":

                if key_threshold_calc not in self._optimal_probabilities.keys():
                    self._optimal_probabilities[key_threshold_calc] = value
                else:
                    self._optimal_probabilities[key_threshold_calc].update(value)
                self._optimal_probabilities[key_threshold_calc]["mean"] = self._calculate_mean(
                    self._optimal_probabilities[key_threshold_calc])

            case "accuracy":

                if key_threshold_calc not in self._acc.keys():
                    self._acc[key_threshold_calc] = value
                else:
                    self._acc[key_threshold_calc].update(value)
                self._acc[key_threshold_calc]["mean"] = self._calculate_mean(self._acc[key_threshold_calc])

            case "sensitivity":
                if key_threshold_calc not in self._sens.keys():
                    self._sens[key_threshold_calc] = value
                else:
                    self._sens[key_threshold_calc].update(value)

                self._sens[key_threshold_calc]["mean"] = self._calculate_mean(self._sens[key_threshold_calc])

            case "specificity":
                if key_threshold_calc not in self._spec.keys():
                    self._spec[key_threshold_calc] = value
                else:
                    self._spec[key_threshold_calc].update(value)
                self._spec[key_threshold_calc]["mean"] = self._calculate_mean(self._spec[key_threshold_calc])

            case "f1":

                if key_threshold_calc not in self._f1.keys():
                    self._f1[key_threshold_calc] = value
                else:
                    self._f1[key_threshold_calc].update(value)
                self._f1[key_threshold_calc]["mean"] = self._calculate_mean(self._f1[key_threshold_calc])

            case "mcc":

                if key_threshold_calc not in self._mcc.keys():
                    self._mcc[key_threshold_calc] = value
                else:
                    self._mcc[key_threshold_calc].update(value)
                self._mcc[key_threshold_calc]["mean"] = self._calculate_mean(self._mcc[key_threshold_calc])

            case "auc_scores_train":

                if key_threshold_calc not in self._auc_scores_training.keys():
                    self._auc_scores_training[key_threshold_calc] = value
                else:
                    self._auc_scores_training[key_threshold_calc].update(value)
                self._auc_scores_training[key_threshold_calc]["mean"] = self._calculate_mean(
                    self._auc_scores_training[key_threshold_calc])

            case "auc_scores_eval":

                if key_threshold_calc not in self._auc_scores_eval.keys():
                    self._auc_scores_eval[key_threshold_calc] = value
                else:
                    self._auc_scores_eval[key_threshold_calc].update(value)
                self._auc_scores_eval[key_threshold_calc]["mean"] = self._calculate_mean(
                    self._auc_scores_eval[key_threshold_calc])

            case "tpr_eval":

                if key_threshold_calc not in self._tprs_eval.keys():
                    self._tprs_eval[key_threshold_calc] = value
                else:
                    self._tprs_eval[key_threshold_calc].update(value)
                self._tprs_eval[key_threshold_calc]["mean"] = self._calculate_mean(self._tprs_eval[key_threshold_calc])

            case "fpr_eval":

                if key_threshold_calc not in self._fprs_eval.keys():
                    self._fprs_eval[key_threshold_calc] = value
                else:
                    self._fprs_eval[key_threshold_calc].update(value)
                self._fprs_eval[key_threshold_calc]["mean"] = self._calculate_mean(self._fprs_eval[key_threshold_calc])

            case "tpr_train":

                if key_threshold_calc not in self._tprs_train.keys():
                    self._tprs_train[key_threshold_calc] = value
                else:
                    self._tprs_train[key_threshold_calc].update(value)
                self._tprs_train[key_threshold_calc]["mean"] = self._calculate_mean(
                    self._tprs_train[key_threshold_calc])

            case "fpr_train":

                if key_threshold_calc not in self._fprs_train.keys():
                    self._fprs_train[key_threshold_calc] = value
                else:
                    self._fprs_train[key_threshold_calc].update(value)
                self._fprs_train[key_threshold_calc]["mean"] = self._calculate_mean(
                    self._fprs_train[key_threshold_calc])

    def update_metrics(self, list_of_metrics: list[GenericMetric]):
        for metric in list_of_metrics:
            name, threshold_calc, metric_dict = metric.to_dict()

            self._set_params(name, threshold_calc, metric_dict)

    @staticmethod
    def _pretty_print_dict(dict_to_print, name_of_dict: str) -> str:

        output_string = "    " + "\"" + name_of_dict + "\"" + ": "
        if len(dict_to_print.items()) > 0:
            output_string += json.dumps(dict_to_print, indent=4)
        else:
            output_string += "\"no_metric_found\""

        return output_string

    def to_json(self):

        if self._metrics_location is None or self._metrics_name is None:
            print("Can not dump metrics")
            return
        else:
            metric_file = self._metrics_location + self._metrics_name + ".json"
            file = open(metric_file, 'w')
            output_string = "{\n"
            output_string += self._pretty_print_dict(self._tprs_train, "tprs_train")
            output_string += ",\n"
            output_string += self._pretty_print_dict(self._fprs_train, "fprs_train")
            output_string += ",\n"
            output_string += self._pretty_print_dict(self._auc_scores_training, "auc_scores_train")
            output_string += ",\n"
            output_string += self._pretty_print_dict(self.tprs_eval, "tprs_eval")
            output_string += ",\n"
            output_string += self._pretty_print_dict(self.fprs_eval, "fprs_eval")
            output_string += ",\n"
            output_string += self._pretty_print_dict(self.auc_scores_eval, "auc_scores_eval")
            output_string += ",\n"
            output_string += self._pretty_print_dict(self._optimal_probabilities, "optimal_probabilities")
            output_string += ",\n"
            output_string += self._pretty_print_dict(self._acc, "accuracy")
            output_string += ",\n"
            output_string += self._pretty_print_dict(self._sens, "sensitivity")
            output_string += ",\n"
            output_string += self._pretty_print_dict(self._spec, "specificity")
            output_string += ",\n"
            output_string += self._pretty_print_dict(self._f1, "f1")
            output_string += ",\n"
            output_string += self._pretty_print_dict(self._mcc, "mcc")
            output_string += "\n"
            output_string += "}"
            file.write(output_string)
            file.close()
            return

    def _calculate_rocs(self):
        used_threshold_calcs = []
        for key, value in self._fprs_eval.items():
            if key in self._tprs_eval.keys():
                used_threshold_calcs.append(key)
        for threshold_calc_name in used_threshold_calcs:
            tprs = self._tprs_eval[threshold_calc_name]
            del tprs["mean"]
            tprs_list = []
            for key, value in tprs.items():
                tprs_list.append(value)

            fprs = self._fprs_eval[threshold_calc_name]
            del fprs["mean"]
            fprs_list = []
            for key, value in fprs.items():
                fprs_list.append(value)
            auc_location = self._metrics_location + self._metrics_name + "_" + threshold_calc_name + ".pdf"
            self._plot_roc_curve(fprs_list, tprs_list, auc_location)

    def _plot_roc_curve(self, fprs, tprs, location):
        """Plot the Receiver Operating Characteristic from a list
        of true positive rates and false positive rates."""
        # Initialize useful lists + the plot axes.
        tprs_interp = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        f, ax = plt.subplots(figsize=(14, 10))

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

        plt.savefig(location, format="pdf")

    def _create_roc_curves(self):
        if self._fprs_eval and self._tprs_eval:
            self._calculate_rocs()
        return

    def save_metrics(self):
        self.to_json()
        self._create_roc_curves()
        return
