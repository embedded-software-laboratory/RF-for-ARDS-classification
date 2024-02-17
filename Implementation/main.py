import json
import math
import sys
from multiprocessing import Pool
import os
import logging



import mysql.connector
import pandas as pd

from ssh import open_ssh
from extraction.MIMICIVExtractor import MIMICIVExtractor
from extraction.eICUExtractor import eICUExtractor
from extraction.ukaExtractor import ukaExtractor

from filtering.MIMICIVFilter import MIMICIVFilter
from filtering.eICUFilter import eICUFilter
from filtering.ukaFilter import ukaFilter

from learning.sk_Learner import SKLearner
from learning.sk_Feature_Selection import SKFeatureSelector

process_connections = {}



"""This file is the starting point of the whole project. The different phases are started using this while."""

def read_options():
    """Reads the options.json file located in the same directory as this file and returns its content"""
    with open("options.json", "r") as file:
        return json.load(file)


def process(options, page_size: int, offset: int, job_id: int) -> list:
    """Extracts and processes the data from a database. The data that is extracted is specified using
     offset and pagesize. job_id is used to identify the process that is started in this function. The 
     return contains the processed data as well as the total frequency of all parameters """
    
    print(f"Starting process {job_id}...")
    
    # Start a new database connection only when the process has not yet started a database connection
    process_id = os.getpid()
    if process_id not in process_connections:
        # Initialize Database connection
        connection = mysql.connector.connect(
            host=options["database_connection"]["host"],
            port=options["database_connection"]["port"],
            user=options["database_connection"]["user"],
            password=options["database_connection"]["password"],
            database=options["database_connection"]["database"],
            use_pure=True,
            ssl_disabled=True
        )
        # Store databse connection in  a global variabel
        process_connections[process_id] = connection

    # Determine right extractor that as specified in the options
    match options["database_connection"]["database"]:
        case "SMITH_MIMICIV":

            extractor = MIMICIVExtractor(options, process_connections[process_id].cursor(), job_id)
        case "SMITH_eICU":

            extractor = eICUExtractor(options, process_connections[process_id].cursor(), job_id)
        case "SMITH_ASIC_SCHEME":

            extractor = ukaExtractor(options, process_connections[process_id].cursor(), job_id)
        case _:
            print("Something went wrong!")
            return
    
    # Set up basic debugging for the extraction and processing
    location = "error_process" + str(job_id) + ".txt"
    logging.basicConfig(filename=location, level=logging.ERROR)
    
    # Create minimal result tupel
    result_tuple=[pd.DataFrame(columns=['Minimaler_Horowitz']), pd.DataFrame(columns=['Minimaler_Horowitz'])]
    try :
        # Extract and process patient data
        patients, freq = extractor.extract_page(page_size, offset)
        print(f"Finished process {job_id}!")
        
        # Create result tuple for return
        result_tuple = [patients, freq]

    except Exception as Argument:
        # Catch exeptions occuring during extraction or processing
        # Due to logging function used no logging of the thrown exception is neccessaray 
        logging.exception(f"Something wrong in {job_id}!")
        raise
    
    return result_tuple


def determine_filtername(dataset_name, path):
    active_filters = options_filtering["parameters"]
    filter_name = ""
    if active_filters["filter_ARDS_high_Horowitz"] == 1 and active_filters["filter_no_ARDS_low_Horowitz"] == 1:
        filter_name = "extreme"
    elif active_filters["filter_ARDS_high_Horowitz"] == 1 and \
            active_filters["filter_no_ARDS_low_Horowitz_contraindication_present"] == 1:
        filter_name = "light"
    elif active_filters["filter_ARDS_high_Horowitz"] == 0 and active_filters["filter_no_ARDS_low_Horowitz"] == 0 and \
            active_filters["filter_no_ARDS_low_Horowitz_contraindication_present"] == 0:
        filter_name = "full"
    else:
        print("Something went wrong!")
        sys.exit(1)
    filtering_path_csv = path + "/" + dataset_name + "_data_" + filter_name + ".csv"
    filtering_path_parquet = path + "/" + dataset_name + "_data_" + filter_name + ".parquet"
    return filtering_path_csv, filtering_path_parquet


def extraction(options, page_size, job_count, location_csv, location_pq, location_latest) -> None:
    """Starts multiple process that extract and process patient data. After processing a file
    containing all data and a file containing the average frequency of parameters are writen to the specified 
    locations"""

    # Initialize variables for processed data and frequencies
    all_results = []
    all_freqs = []

    # Start a process pool with multiple process that run in parallel to extract and preprocess the data
    with Pool(processes=12) as pool:
        result = pool.starmap(process, [(options, page_size, i * page_size, i) for i in range(job_count)])

    # Append frequency and processed data to the right variable
    for item in result:
        all_results.append(item[0])
        all_freqs.append(item[1])

    # Put different processed data frames into a single data frame and ensure a contiounus index
    data = pd.concat(all_results)
    data= data.reset_index()
    del data['index']

    # Calculate frequencies of parameters per day
    freq = pd.concat(all_freqs).sum().divide(len(data.index)*14)

    # Write extracted data to files
    freq.to_csv("../Data/Databases/freq.txt")
    write_latest_stage(data, location_latest, location_pq, location_csv)


def filtering(filter_data, location_pq, location_filtered_csv, location_filtered_pq, location_latest) -> None:
    """Reads the specified in location_pq before applying different selectable filters to it. 
    Lastly the filtered data is written to a  file. The filter class to be used is specified in 
    filter_data"""

    # Read data and ensure continous index
    data = pd.read_parquet(location_pq, engine='auto')
    data = data.reset_index()
    del data['index']

    # Apply filter
    data_filtered = filter_data.filter_admissions_learning(data)
    
    # Write filtered data to file
    write_latest_stage(data_filtered, location_latest, location_filtered_pq, location_filtered_csv)
    return


def write_latest_stage(data: pd.DataFrame, location_latest, location_parquet, location_csv) -> None:
    """Writes data frames to specified locations"""
    
    # Write data into a file that is used in the next step
    data.to_parquet(location_latest, engine='auto', compression='snappy', index=None)
    
    # Write data to a file that is NOT overwritten in the next step
    data.to_parquet(location_parquet, engine='auto', compression='snappy', index=None)
    data.to_csv(location_csv, sep=",", index=False)

if __name__ == "__main__":
    """Main function of the programm: Initializes needed variables and controls execution flow 
    according to the specified options"""
    
    # Read options and store them in variables
    print("Reading options")
    options = read_options()
    options_execution = options["execution"]
    options_used_datasets = options_execution["datasets"]
    active_phases = options_execution["active_phases"]
    options_extraction = options["extraction"]
    options_filtering = options["filtering"]
    options_learning = options["learning"]

    # Controll execution flow

    # Check if extraction flag is active
    if active_phases["extraction"] == 1:
        print("Connecting to SSH")
        # Intialize ssh connection
        tunnel = open_ssh(options["database_connection"]["ssh"]["host"],
                          options["database_connection"]["ssh"]["username"],
                          options["database_connection"]["ssh"]["password"],
                          options["database_connection"]["ssh"]["forward_host"],
                          options["database_connection"]["ssh"]["forward_port"])
        options["database_connection"]["host"] = "localhost"  # This makes extractors use the SSH tunnel
        options["database_connection"]["port"] = tunnel.local_bind_port
        page_size = options_extraction["parameters"]["page_size"]

        # Determine which databases are flagged for extraction
        if options_extraction["used_databases"]["uka"] == 1:
            
            # Set connnection parameter
            options["database_connection"]["database"] = "SMITH_ASIC_SCHEME"

            # Calculate number of jobs needed for extracting all data
            admission_count = options_extraction["parameters"]["admission_count_uka"]
            job_count = math.ceil(admission_count / page_size)

            # Start extraction and processing of data
            print(f"Starting {job_count} processes for extraction of uka data...")
            extraction(options, page_size, job_count, options_extraction["locations"]["uka_csv"],
                       options_extraction["locations"]["uka_pq"], options_execution["locations"]["uka"])

        if options_extraction["used_databases"]["MIMICIV"] == 1:

            # Set connection parameter
            options["database_connection"]["database"] = "SMITH_MIMICIV"

            # Calculate number of jobs needed for extracting all data
            admission_count = options_extraction["parameters"]["admission_count_MIMICIV"]
            job_count = math.ceil(admission_count / page_size)

            # Extract and process data
            print(f"Starting {job_count} processes for extraction of MIMIC IV data...")
            extraction(options, page_size, job_count, options_extraction["locations"]["MIMICIV_csv"],
                       options_extraction["locations"]["MIMICIV_pq"], options_execution["locations"]["MIMICIV"])
        
        if options_extraction["used_databases"]["eICU"] == 1:
            
            # Set connection parameter
            options["database_connection"]["database"] = "SMITH_eICU"

            # Calculate number of process needed to extract all data
            admission_count = options_extraction["parameters"]["admission_count_eICU"]
            job_count = math.ceil(admission_count / page_size)
            

            # Extract and process data
            print(f"Starting {job_count} processes for extraction of eICU data...")
            extraction(options, page_size, job_count, options_extraction["locations"]["eICU_csv"],
                       options_extraction["locations"]["eICU_pq"], options_execution["locations"]["eICU"])

    # Check if filtering flag is set
    if active_phases["filtering"] == 1:
        # Create correct filter for the databases
        path_to_filtered = options_filtering["locations"]["general"]
        if options_used_datasets["uka"] == 1:
            filter_data = ukaFilter(options)
            location_filtered_csv, location_filtered_parquet = determine_filtername("uka", path_to_filtered)
            # Start filtering
            filtering(filter_data, options_extraction["locations"]["uka_pq"], location_filtered_csv,
                      location_filtered_parquet, options_execution["locations"]["uka"])
        if options_used_datasets["eICU"] == 1:
            filter_data = eICUFilter(options)
            location_filtered_csv, location_filtered_parquet = determine_filtername("eICU", path_to_filtered)
            # Start filtering
            filtering(filter_data, options_extraction["locations"]["eICU_pq"], location_filtered_csv,
                      location_filtered_parquet, options_execution["locations"]["eICU"])
        if options_used_datasets["MIMICIV"] == 1:
            filter_data = MIMICIVFilter(options)
            location_filtered_csv, location_filtered_parquet = determine_filtername("MIMICIV", path_to_filtered)
            # Start filtering
            filtering(filter_data, options_extraction["locations"]["MIMICIV_pq"],
                      location_filtered_csv, location_filtered_parquet,
                      options_execution["locations"]["MIMICIV"])

    # Check if feature selection flag is active and start feature selection
    if active_phases["feature_selection"] == 1:
        selector = SKFeatureSelector(options)
        selector.feature_selection()

    # Check if cross validation flag is active and start cross validation
    if active_phases["cross_validation"] == 1:
        if options_learning["algorithm"] == "sk":
            learner = SKLearner(options)
            learner._cross_validate_forest()
        else:
            print("Unknown learning library")

    # Check if learning flag is active and start  learning
    if active_phases["learning"] == 1:
        if options_learning["algorithm"] == "sk":
            learner = SKLearner(options)
            learner._learn()
        else:
            print("Unknown learning library")

    # Check if evaluation flag is active and start evaluating
    if active_phases["evaluation"] == 1:
        if options_learning["algorithm"] == "sk":

            learner = SKLearner(options)
            learner.evaluate()
        
