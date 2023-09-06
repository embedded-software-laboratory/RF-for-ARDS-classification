import pandas as pd

databases = ["uka", "eICU", "MIMICIV"]
filters = ["full", "light", "extreme"]

for db in databases:
    for f in filters:
        name = db + "_data_" + f + ".parquet"
        location = "../../Data/Processed_Data/"+name
        data = pd.read_parquet(location, engine="auto")
        ards = data.query('ARDS == 1')
        no_ards = data.query('ARDS == 0')
        print(name)
        print("No ARDS: " + str(len(no_ards.index)))
        print("ARDS: " + str(len(ards.index)))
        print("TOTAL: " + str(len(no_ards.index)+len(ards.index)))