# Clemence Le Cornec
# 10/07/2018

from utils import *
import os

# DfT
fileDfT = os.getcwd() + "\\Data\\"

listOfFiles = [f for f in os.listdir(path + "Training\\")]
print(listOfFiles)

columns = ['Test Time', 'GPS Speed', 'Tailpipe1 Exh. Flow Rate(Std. 20 deg C)', 'Tailpipe NOx Delayed Conc.']

data = load_DfT_data(path = fileDfT, file_name = listOfFiles[0], sheetName = 'R-Combined',\
					skipRows = 0, skipFooter = 0)

print(data)

emissions = computeEmissionsInGPerS(data['Tailpipe NOx Delayed Conc.'], data['Tailpipe1 Exh. Flow Rate(Std. 20 deg C)'])

print(emissions)