#[[[cog
#import cog
#import os
#from os.path import exists
#
#lines = []
#
#with open("../../Data/uka_data_column_names.txt","r") as file:
#   lines = [line.rstrip().replace('\r','').replace('\n','') for line in file]
#   
#   
#    
#for i in range(len(lines)):
#   
#   cog.outl("SELECT COUNT(%s)/(MAX(ud.Zeit_ab_Aufnahme)/1440) \n FROM SMITH_ASIC_SCHEME.uka_data ud GROUP BY patientid;" % (lines[i]))
#]]]
#[[[end]]]