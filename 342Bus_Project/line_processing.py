# +
import numpy as np
import pandas as pd
import os

BusFile = os.getcwd() + "/342Bus/Source/Lines.csv"
BusFile = BusFile.replace("\\","/")
BusFile = BusFile.replace('/342Bus/342Bus/', '/342Bus/')
print(BusFile)
lines = pd.read_csv(BusFile, skiprows=1)
lines
# lines_list = []
# for l in lines:
#     # print(l)
#     e = l.split()[1:5]
#     e.append(l.split()[-1])
#     print(e)
#     lines_list.append(e)
