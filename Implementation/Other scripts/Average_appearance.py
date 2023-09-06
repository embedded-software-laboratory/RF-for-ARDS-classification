"""Script to determine rate of appearance of each paramter over all databases"""

import pandas as pd
from functools import reduce
values = []
data_eICU = pd.read_csv("../../Data/Databases/freq.csv", sep=",")
data_MIMICIV = pd.read_csv("../../Data/Databases/freq_Mimiciv.csv", sep=",")
data_eICU['per h'] = data_eICU['0']/24
data_eICU = data_eICU.rename(columns={'Unnamed: 0' : 'name', '0' : 'presence'})
data_eICU.drop(0, inplace=True)
data_MIMICIV['0'] = data_MIMICIV['0']/14
data_MIMICIV['per h'] = data_MIMICIV['0']/(24)
data_MIMICIV = data_MIMICIV.rename(columns={'Unnamed: 0' : 'name', '0' : 'presence'})
data_MIMICIV.drop(0, inplace=True)


with open("../../Data/Databases/avg_uka.txt", "r") as file_column_names:
    values = [float(line.rstrip().replace('\r','').replace('\n','')) for line in file_column_names]
data_uka = pd.DataFrame(columns=['name', 'presence', 'per h'])
data_uka['name'] = data_eICU['name']
data_uka['presence'] = values
data_uka['per h'] = data_uka['presence']/24

list_data = [data_uka, data_eICU, data_MIMICIV]
data = reduce(lambda  left,right: pd.merge(left,right,on=['name'], how='inner'), list_data)

data['max'] = data[['per h_x','per h_y','per h']].max(axis=1)
list_intervall = []
list_intervall_uk = []
intervall_group = []
for _, row in data.iterrows():
    intervall = (1/row['max'])*60
    intervall_uk = intervall/4
    list_intervall.append(intervall)
    list_intervall_uk.append(intervall_uk)
    if intervall <= 60:
        intervall_group.append("hourly")
    elif intervall <= 120 : 
        intervall_group.append("every 2 hours")
    elif intervall <= 240:
        intervall_group.append("every 4 hours")
    elif intervall <= 480 : 
        intervall_group.append("every 8 hours")
    elif intervall <= 720 : 
        intervall_group.append("two times a day")
    elif intervall <= 1440 : 
        intervall_group.append("daily")
    elif intervall <= 2880 : 
        intervall_group.append("every other day")
    elif intervall<=7200 :
        intervall_group.append("twice")
    else :
        intervall_group.append("once")
data['uka'] = list_intervall_uk
data['other'] = list_intervall
data['sampling'] = intervall_group
work = pd.DataFrame(columns=['name', 'max avagerage number of measurements per h', 'sampling rate'])
work['name'] = data['name']
work['max avagerage number of measurements per h'] = data['max']
work['sampling rate'] = data['sampling']
work.to_csv("./test.csv", sep=",", index=False)
print(data.to_string())
print(work.to_string())
work['max avagerage number of measurements per h'].to_clipboard(index=False)


