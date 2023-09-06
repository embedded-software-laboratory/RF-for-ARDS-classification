from Lookup.features_new import *

import pandas as pd
import numpy as np

list_hourly = list_features_names[-1]
list_two_h = list_features_names[-2]
list_hourly_params = [[] for _ in range(len(list_hourly))]
list_bihourly_params = [[] for _ in range(len(list_two_h))]
print(len(list_hourly_params))
print(len(list_hourly))

data = pd.read_parquet("./uka.parquet", engine="auto")

for index in range(len(list_hourly)):
    for column in data:
        if list_hourly[index] in column:
            if list_hourly[index] == 'AF' and 'AF_spontan' in column:
                continue
            list_hourly_params[index].append(column)

for index in range(len(list_two_h)):
    for column in data:
        if list_two_h[index] in column:
            list_bihourly_params[index].append(column)
to_drop = []
for index in range(len(list_hourly)) :
    print(index)
    data_entry = data[list_hourly_params[index]]
    grouped_list = [[] for _ in range(0,60,1)]
    for i, row in data_entry.iterrows() :
        grouped = row.groupby(np.arange(len(list_hourly_params[index]))//4).mean()
        for j in range(grouped.size):
            grouped_list[j].append(grouped.at[j])
    param = list_hourly[index]
    for i in range(len(grouped_list)):
        feature = param + str(i)
        data[feature] = grouped_list[i]

        
            
    
    for i in range(60,240,1):
        to_drop.append(list_hourly[index] + str(i))
print(to_drop)

for index in range(len(list_two_h)) :
    print(index)
    data_entry = data[list_bihourly_params[index]]
    grouped_list = [[] for _ in range(0,60,1)]
    for i, row in data_entry.iterrows() :
        grouped = row.groupby(np.arange(len(list_bihourly_params[index]))//2).mean()
        for j in range(grouped.size):
            grouped_list[j].append(grouped.at[j])
    param = list_two_h[index]
    for i in range(len(grouped_list)):
        feature = param + str(i)
        data[feature] = grouped_list[i]

        
            
    
    for i in range(60,120,1):
        to_drop.append(list_two_h[index] + str(i))
data.drop(to_drop, axis=1, inplace=True)
data = data.reset_index()
del data['index']
data.to_parquet("./save.parquet", engine="auto", compression='snappy', index=False)
    

        
         