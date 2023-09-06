#[[[cog
#import cog
#import os
#from os.path import exists
#
#lines = []
#
#with open("uka.sql","r") as file:
#   lines = [line.rstrip().replace('\r','').replace('\n','') for line in file]
#   lines = [line[:-1] for line in lines]
#   
#   
#    
#for i in range(len(lines)):
#   
#   cog.outl("SELECT SUM(a.avg)/13067 as oavg FROM (%s) as a ;" % (lines[i]))
#]]]
#[[[end]]]