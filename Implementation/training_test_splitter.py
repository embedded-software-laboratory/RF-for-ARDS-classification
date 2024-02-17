import random
import math

import pandas as pd

def _generate_new_full_dataset(data) -> None :
        data = data.fillna(-100000)
        data = data.reset_index()
        del data['index']
        data_ards = data.query("ARDS == 1").reset_index()
        del data_ards["index"]
        data_no_ards = data.query("ARDS == 0").reset_index()
        
        del data_no_ards["index"]
        num_ards_real = len(data_ards.index)
        num_no_ards_real = len(data_no_ards.index)
        total_calculated_ards = math.ceil(num_ards_real*(1/0.1))
        num_data_no_ards_target = total_calculated_ards - num_ards_real
        total_calculated_no_ards = math.ceil(num_no_ards_real*(1/(1-0.1)))
        num_data_ards_target = total_calculated_no_ards-num_no_ards_real
        if num_data_no_ards_target > num_no_ards_real:
            list_ards_indicies = random.sample(range(num_ards_real), num_data_ards_target)
            data_ards = data_ards.iloc[list_ards_indicies].reset_index()
            del data_ards['index']
        else :
            list_no_ards_indicies = random.sample(range(num_no_ards_real), num_data_no_ards_target)
            data_no_ards = data_no_ards.iloc[list_no_ards_indicies].reset_index()
            del data_no_ards["index"]
        return _generate_split_dataset(data_ards, data_no_ards)


def _generate_split_dataset(data_ards: pd.DataFrame, data_no_ards: pd.DataFrame) -> None:
        num_ards = len(data_ards.index)
        num_no_ards = len(data_no_ards.index)
        print("ARDS " + str(num_ards))
        print("NO ARDS: " + str(num_no_ards))
        print("TOTAL: " + str(num_ards+num_no_ards))
        num_ards_test = math.floor(num_ards*0.25)
        num_no_ards_test = math.floor(num_no_ards*0.25)
        list_indicies_ards_test = random.sample(range(num_ards), num_ards_test)
        list_indicies_no_ards_test = random.sample(range(num_no_ards), num_no_ards_test)
        data_ards_test = data_ards.iloc[list_indicies_ards_test].reset_index()
        del data_ards_test["index"]
        data_no_ards_test = data_no_ards.iloc[list_indicies_no_ards_test].reset_index()
        del data_no_ards_test["index"]
        test = pd.concat([data_no_ards_test, data_ards_test])
        data_ards_training = data_ards[~data_ards.index.isin(list_indicies_ards_test)]
        data_no_ards_training = data_no_ards[~data_no_ards.index.isin(list_indicies_no_ards_test)]
        training = pd.concat([data_no_ards_training, data_ards_training])
        print(len(training.index))
        print(len(test.index))
        return training, test
    
def _write_data(data: pd.DataFrame, location) :
        location_pq = location + ".parquet"
        location_csv = location + ".csv"
        data.to_parquet(location_pq, engine='auto', compression='snappy', index=None)
        data.to_csv(location_csv, sep=",", index=None)

def _delete_admissions(data: pd.DataFrame, to_delete: list) -> pd.DataFrame:
    to_delete_set = set(to_delete)
    data = data.drop(to_delete_set)
    data.reset_index(inplace= True)
    del data['index']
    return data

def filter_admissions_learning(data:pd.DataFrame, high, low, contra):
    to_delete = []
    counter_High_ARDS = 0
    counter_Low_No_ARDS = 0
    counter_Low_No_Contra = 0
    for index, row in data.iterrows():
        if high == 1:
            if row['ARDS'] == 1 and row['Minimaler_Horowitz'] > 300:
                to_delete.append(index)
                counter_High_ARDS = counter_High_ARDS + 1
        if low == 1:
            if row["ARDS"] == 0 and row["Minimaler_Horowitz"] < 200:
                to_delete.append(index)
                counter_Low_No_ARDS = counter_Low_No_ARDS +1
        elif contra == 1:
            contraindidaction = 1 if (row["Lungenoedem"] == 1 or row["Herzinsuffizienz"] == 1 or row["Hypervolaemie"] == 1) else 0
            if contraindidaction == 0 and row["ARDS"] == 0 and row["Minimaler_Horowitz"] < 200:
                to_delete.append(index)
                counter_Low_No_Contra = counter_Low_No_Contra +1
            
                
        
    print("HIGH ARDS:" + str(counter_High_ARDS))
    print("LOW NO ARDS: " + str(counter_Low_No_ARDS))
    print("LOW NO ARDS NO Contra: " + str(counter_Low_No_Contra))
    print("Before :" + str(len(data.index)))
    data_filtered = _delete_admissions(data, to_delete)
    print("After :" + str(len(data_filtered.index)))                
        
        


    return data_filtered

   


if __name__ == '__main__' :
    databases = [ "uka", "eICU", "MIMICIV"]
    filters = ["full", "extreme", "light"]
    for db in databases:
       
        for f in filters:
            file = "../Data/Extracted_Data/" + db + "_data.parquet"
            data = pd.read_parquet(file, engine='auto')
            
            
            location = db + "_data_" + f
            print(location)
            location_training = "../Data/Training_Data/" + location 
            test_location = "../Data/Test_Data/" + location 
            if f == "extreme" :
                data = filter_admissions_learning(data, 1, 1, 0)

            elif f == "light" :
                data = filter_admissions_learning(data, 1, 0, 1)
            else :
                data = data
            print(data.shape)
            training, test = _generate_new_full_dataset(data)
            print(training.shape)
            _write_data(test, test_location)
            _write_data(training, location_training)


