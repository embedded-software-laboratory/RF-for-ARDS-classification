from Implementation.Lookup.all_features import list_all_features
import pandas as pd

for db in ['uka', 'eICU', 'MIMICIV']:
    for filtering in ['full', 'light', 'extreme']:
        name = db + '_data_' + filtering + '.parquet'
        data = pd.read_parquet("../../Data/Training_Data/" + name, engine='auto')
        set_features = set(list_all_features)
        print(name)
        for col in data.columns:
            if col not in set_features:
                print("Extra Feature: " + str(col))
            else:
                set_features.remove(col)
        print(set_features)
