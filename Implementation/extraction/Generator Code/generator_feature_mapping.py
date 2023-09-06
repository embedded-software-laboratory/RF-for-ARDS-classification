#[[[cog
#import cog
#import os
#
#lines = []
#with open("../../../Data/Databases/variables.txt","r") as file:
#    lines = file.readlines()
#    lines = [line.rstrip() for line in lines]
#cog.outl("feature_map_dict = {")
#for i in range(len(lines)):
#    if i+1 == len(lines):
#        cog.outl("'%s' : %s" % (lines[i], i) )
#    else:
#        cog.outl("'%s' : %s," % (lines[i], i) )
#cog.outl("}")
#cog.outl("feature_map_dict_rev = {")
#for i in range(len(lines)):
#    if i+1 == len(lines):
#        cog.outl("%s : '%s'" % (i, lines[i]) )
#    else:
#        cog.outl("%s : '%s'," % (i, lines[i]) )
#cog.outl("}")
#cog.outl("list_all_features = [")
#for i in range(len(lines)):
#    if i+1 == len(lines):
#        cog.outl("'%s'" % lines[i] )
#    else:
#        cog.outl("'%s'," %  lines[i] )
#cog.outl("]")
#]]]
#[[[end]]]