This folder contains my implementation.

# Folder structure

* Extraction contains the different Extractors used for the different databases as well as generator code used to create several dictionaries. IExtractor contains functions used by all extractors
* Lookup contains said dictionaries as well es other files which store further information needed for the execution of my implementation
* Learning contains all files where a RF is used in some way as well as part of the evaluation code
* Filtering contains code that is used to filter the data during different steps of the program execution. Filters specific to database are mostly found in their respective filter file. IFilter contains general filters.
* Preprocessing contains the code that transformed the extracted data to feature vectors. Each database has their own Preprocessor with functions specific to their structure, common functions are found in IPreprocessor.
* ssh contains code to establish an SSH connection to the database server.
* Other scripts contain a multitude of scripts used sometime during my thesis
* eICU.parquet, uka.parquet and MIMIC-IV.parquet should contain the data after preprocessing
* main.py is the main file used for execution, other .py files are similar to the ones in other scripts but need to stay at their current location in order to work

# Usage of options.json
WARNING use linux style paths for each path provided in the options.json.

## Execution
Contains settings for the execution flow of the program

* active_phases notes which phases of the program are executed. 1 denotes that the respective phase should be executed
* datasets denotes which data is relevant for the filters during a run of the program and to be used in learning, cross-validation and evaluation. All relevant databases should be marked by 1
* locations store where the current (processed) data will be stored. This file will be overwritten multiple times during program execution

## Extraction
Contains settings for the extraction phase with regard to the databases

* used_databases marks whether data should be extracted from the respective databases (1: yes, 0: no).
* locations store the location where the extracted data will be stored as a .csv and .parquet file.
* parameters contains multiple settings for the extraction itself:
    * admission_count_database : How many admissions will be extracted from the database
    * page_size: How many admissions will be extracted in each job

## database_connection
Contains settings needed to establish a connection to the database and ssh server for i11

* user : database username
* password: database password
* ssh : Contains information on the ssh connection as well as forwarding
    * host: ip-address of the ssh-server
    * username: ssh-username
    * password: ssh-password
    * forward-host: ip-address the ssh-server will forward your connection to
    * forward-port: port where your connection will be forwarded to

## extraction_parameter
Contains parameters for the processing of extracted data.

* windowsize_before: Number of days for which data is extracted before the timestamp of lowest horovitz-index
* windowsize_after: Number of days for which data is extracted after the timestamp of lowest horovitz-index
* windowsize_horowitz: Number of horovitz-values considered during rolling window averaging of horovitz-index
* cutoff_horowitz: Maximum time between two horovitz-index values before a new window is started
* cutoff_time_fiO2_paO2: not used anymore
* cutoff_time_lymphs_leukos: Maximum time allowed between values considered for calculation of lymphocyte related parameters
* feature_cutoff: Borders for feature frequency (feature must have at least that many occurences per day in order to belong to this group), number of values = number of feature groups, not limited in size

## Filtering
Contains parameters used during the filtering

* parameters: 
    * filter_ARDS_high_Horowitz: if 1 admissions with ARDS with a lowest horovitz-index > 300 are filtered out
    * filter_no_ARDS_low_Horowitz: if 1 admissions without ARDS with a lowest horovitz-index < 200 are filtered out
    * filter_no_ARDS_low_Horowitz_contraindication_present: if 1 admissions without ARDS with a lowest horovitz-index < 200 while not having a contra indication diagnosis (lung edema, heart failure, etc.) are filtered out
    * filter_percentage_of_features: admissions need at least this percentage value of all features present. Use decimal notation
    * filter_features_present: If 1 ensure at least filter_percentage_of_features are present for the admission, else it gets filtered out
* locations: path to the different locations were the filtered data is stored in .csv and .parquet files

## feature_selection
* params:
    * type: How to Select the best Features, either percentage based (SelectPercentile) or absolute (SelectKBest) incase of algorithm mutual_info_classif or RandomForest in case of algorithm Tree
    * algorithms: Name of algorithm used for feature selection Possible options Tree or mutual_info_classif
    * generate_new_sets: Make a new split of the extracted data (1: yes, 0: no)
* algorithm_params : Parameters for feature selection
    * k: Number of absolute features chosen
    * percentile : percentage of features chosen
    * Random_Forest_location : path to configuration file for Random Forest

## Learning
Contains parameters for learning and evaluation

* parameters :
    * general : 
        * model_name : Name of your model, will be used when storing or loading RF models
        * training_uka_database: Whether or not data from this database will be used during training
        * ratio_ards_no_ards: Desired percentage of ARDS admissions in both Test and validation set. Will be achieved by lowering admissions used for training
        * ratio_test_training: Percentage of admissions reserved for the validation set
        * generate_new_sets: Make a new split of the extracted data (1: yes, 0: no) only set to one if generate_new_sets in feature selection is set to 0
    * sk-learn: 
    * random_forest: parameters to configure the RF explanation found [here](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html). All config files for RF must have this structure
    * cross_validation : explained [here](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold) RandomState is hardcoded.

* locations:
    * test_set_location : Location of validation set (.parquet file extension not needed), model_name will be used to complete path
    * training_set_location: Location of training set (.parquet file, file extension not needed), model_name will be used to complete path
    * training_set_location_csv : Location of training set (.csv file)
    * validation_options_name : name of the model to validate
    * validation_options_path : Location of models used for validation, validation_options_name will be used to complete path
    * validation_metrics_path : Location where the metrics of model with name validation_options_name will be stored,  validation_options_name will be used to complete path
* algorithm: Type of RF algorithm used for learning only sk-learn (sk) is implemented 

# evaluation

Not used as it got incorporated in learning

#