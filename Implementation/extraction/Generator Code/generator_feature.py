#[[[cog
#import cog
#import os
#import json
#import math
#import sys
#
#lines_avg_values = []
#line_names = []
#list_windowsizes = [[],[],[]]
#options = {}
#with open("../../options.json") as jsonfile:
#    options = json.load(jsonfile)
#feature_cutoff = options["extraction_parameter"]["feature_parameter"]["feature_cutoff"]
#windowsize_data = options["extraction_parameter"]["windowsize_before"] + options["extraction_parameter"]["windowsize_after"]
#with open("../../../Data/Databases/variables.txt", "r") as file_column_names:
#    line_names = [line.rstrip().replace('\r','').replace('\n','') for line in file_column_names]
#with open("../../../Data/Databases/test_avg.txt","r") as file_avg_values:
#    lines_avg_values = [float(line.rstrip().replace('\r','').replace('\n','')) for line in file_avg_values]
#del line_names[0]
#feature_category_list_names = [[] for i in range(len(feature_cutoff)+1)]
#feature_category_list_var = [[] for i in range(len(feature_cutoff)+1)]
#feature_cutoff.append(0.0)
#feature_cutoff.append(float("inf"))
#feature_cutoff.sort()
#print(len(feature_cutoff))
#for i in range(len(feature_cutoff)-1):
#    try:
#        windowsize = 24*4/feature_cutoff[i]
#    except:
#        windowsize = 0
#    list_windowsizes[0].append(windowsize)
#    try:
#        windowsize = 24*60/feature_cutoff[i]
#    except:
#       windowsize = 0
#    list_windowsizes[1].append(windowsize)
#    list_windowsizes[2].append(windowsize)
#for i in range(len(feature_cutoff)-1):
#    cutoff = feature_cutoff[i+1]
#    last_cutoff = feature_cutoff[i]
#    print(cutoff)
#    for j in range(len(lines_avg_values)):
#        
#        if (lines_avg_values[j]*24) < cutoff and not float(lines_avg_values[j]*24) < last_cutoff :
#            print(line_names[j])
#            feature_category_list_names[i].append(line_names[j])
#cog.outl("list_features_names=[")
#counter = 0
#for feature_category in feature_category_list_names:
#    if counter+1==len(feature_category_list_names):
#        cog.outl("%s" % feature_category)       
#    else :
#        cog.outl("%s," % feature_category)
#        counter+=1
#
#cog.outl("]")
#cog.outl("feature_position_dict_uka = {")
#windowsizes_uka = list_windowsizes[0]
#for i in range(len(feature_category_list_names)):
#    windowsize_feature = windowsizes_uka[i]
#    for j in range(len(feature_category_list_names[i])):
#        
#        if j+1 == len(feature_category_list_names[i]) and i+1 == len(feature_category_list_names):
#            cog.outl("'%s': %s" % (feature_category_list_names[i][j], windowsize_feature))
#        else :
#            cog.outl("'%s': %s," % (feature_category_list_names[i][j], windowsize_feature))
#cog.outl("}")
#cog.outl("feature_position_dict_eICU_MIMIC = {")
#windowsizes_uka = list_windowsizes[1]
#for i in range(len(feature_category_list_names)):
#    windowsize_feature = windowsizes_uka[i]
#    for j in range(len(feature_category_list_names[i])):
#        
#        if j+1 == len(feature_category_list_names[i]) and i+1 == len(feature_category_list_names):
#            cog.outl("'%s': %s" % (feature_category_list_names[i][j], windowsize_feature))
#        else :
#            cog.outl("'%s': %s," % (feature_category_list_names[i][j], windowsize_feature))
# 
#cog.outl("}")
#del feature_cutoff[-1]
#number_of_times = 0
#for i in range(len(feature_category_list_names)):
#    if i==0 :
#        number_of_times = 1
#    else :
#        number_of_times = math.ceil(windowsize_data*feature_cutoff[i])
#    for j in range(len(feature_category_list_names[i])):
#        for h in range(number_of_times):
#            var_string = feature_category_list_names[i][j] + str(h) if not number_of_times == 1 else feature_category_list_names[i][j]
#            
#            feature_category_list_var[i].append(var_string)
#
#cog.outl("list_features_var=[")
#for i in range(len(feature_category_list_var)):
#    for j in range(len(feature_category_list_var[i])):
#        if i+1==len(feature_category_list_var) and j+1==len(feature_category_list_var[i]):
#            cog.outl("'%s'" % feature_category_list_var[i][j])
#        else:
#            cog.outl("'%s'," % feature_category_list_var[i][j])
#cog.outl("]")
#var_index = 0
#cog.outl("dict_features_var= {")
#for i in range(len(feature_category_list_var)):
#    for j in range(len(feature_category_list_var[i])):
#        if i+1==len(feature_category_list_var) and j+1==len(feature_category_list_var[i]):
#            cog.outl(" %s : '%s'" % (var_index, feature_category_list_var[i][j]))
#            var_index+=1
#        else:
#            cog.outl(" %s : '%s'," % (var_index, feature_category_list_var[i][j]))
#            var_index+=1
#cog.outl("}")
#cog.outl("list_all_feature_names = [")
#for i in range(len(feature_category_list_names)):
#    for j in range(len(feature_category_list_names[i])):
#        if j+1 == len(feature_category_list_names[i]) and i+1 == len(feature_category_list_names):
#            cog.outl("'%s'" % feature_category_list_names[i][j])
#        else :
#            cog.outl("'%s'," % feature_category_list_names[i][j])
#cog.outl("]")        
#cog.outl("list_windowsizes= [")
#counter = 0
#for window in list_windowsizes:
#    if counter+1==len(list_windowsizes):
#        cog.outl("%s" % window)       
#    else :
#        cog.outl("%s," % window)
#        counter+=1
#cog.outl("]")
#]]]
#[[[end]]]