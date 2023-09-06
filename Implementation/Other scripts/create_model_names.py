import pandas as pd
"""Script used to create all needed model names"""
databases = ["eICU", "uka", "MIMIC-IV"]
filters = ["AC", "BC", "None"]
featureselection = ["full", "60", "30", "RF"]

names = []
featureselection_list = []
filters_list = []
dbs = []
for db in databases:
    for f in filters:
        for fs in featureselection:
            name = db + "-" + f + "-" + fs
            alg = ""
            fil = ""
            match f :
                case "AC" :
                    fil = "a,c"
                case "BC" :
                    fil ="b,c"
                case "None" :
                    fil = "none"
            match fs:
                case "60":
                    alg = "mutual information 60%"
                case "30":
                    alg = "mutual information 30%"
                
                case "RF":
                    alg = "Random Forest"
                
                case "full":
                    alg = "none"
            names.append(name)
            featureselection_list.append(alg)
            filters_list.append(fil)
            dbs.append(db)
data = pd.DataFrame(columns=["Modelname", "Database", "Filters used", "Type of feature selection"])
data["Modelname"] = names
data["Database"] = dbs
data["Filters used"] = filters_list
data["Type of feature selection"] = featureselection_list
print(data.to_string())
data.to_csv("./Models.csv", sep=",", index=False)


           
            