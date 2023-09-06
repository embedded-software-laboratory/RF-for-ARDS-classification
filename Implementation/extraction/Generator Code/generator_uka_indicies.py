#[[[cog
#import cog
#import os
#from os.path import exists
#
#lines = []
#if exists('uka_data_indicies.py'):
#   os.remove('uka_data_indicies.py')
#with open("../../Data/uka_data_column_names.txt","r") as file:
#   lines = [line.rstrip() for line in file]
#   
#   lines = [line.replace("-", "") for line in lines]
#   lines = [line.replace("(", "") for line in lines]
#   lines = [line.replace(")", "") for line in lines]
#   lines = [line.upper() for line in lines]
#    
#for i in range(len(lines)):
#    cog.outl("%s = %s" % (lines[i], i) )
#]]]
#[[[end]]]